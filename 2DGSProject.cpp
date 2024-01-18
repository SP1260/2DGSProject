
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

// Used for loss calc
#include <torch/script.h>
#include <torch/nn/modules/loss.h>
#include <torch/nn/functional.h>

#include <opencv2/highgui/highgui.hpp>

//struct Splat {
//	torch::Tensor position;  // 2D position vector
//	torch::Tensor covarianceMatrix;  // 2x2 covariance matrix
//	torch::Tensor color;  // RGB color
//	float alpha;  // Alpha value
//};

class SplatRenderer : public torch::nn::Module {

private:
	int width;
	int height;
	torch::Tensor splatParams;
	int splatSize;
	int iterations;//number of optimization iterations
	int refineIterations;
	torch::Device device;

public:
	SplatRenderer(int width, int height, torch::Tensor& params, const torch::DeviceType &dev) : width(width), height(height), device(dev) {
		//register parameters for the training
		splatParams = register_parameter("splatParams", params, true);

		// Se the splat size to be a portion of the image size
		splatSize = int(0.45*std::min(width, height));

		iterations = 1000;
		refineIterations = iterations / 10;//150;		
	}	

	// Define the optimization step
	void optimize(const torch::Tensor& targetImage, torch::optim::Adam& optimizer, int numSplats, int numInitSplats) {
		double alphaThreshold = 0.01;
		double gradientThreshold = 0.002;
		double gaussianThreshold = 0.002;		
		int relevantSplatsEnd = numInitSplats;
		int storeIterations = 15;

		//Mask the relevant splats
		torch::Tensor relevantSplatMask = torch::cat({ torch::ones({numInitSplats}, torch::kBool), torch::zeros({numSplats - numInitSplats}, torch::kBool) }, 0).to(device);

		//cv::namedWindow("Screen", cv::WINDOW_AUTOSIZE);

		for (int iter = 0; iter < iterations; ++iter) {		

			//Forward pass
			torch::Tensor relevantSplats = splatParams.index({ relevantSplatMask }).to(device);
			
			int relevantSplatsNum = relevantSplats.size(0);
			
			auto scaleX = torch::sigmoid(relevantSplats.index({ torch::indexing::Slice(), 0 })).to(device);
			auto scaleY = torch::sigmoid(relevantSplats.index({ torch::indexing::Slice(), 1 })).to(device);
			auto theta = torch::sigmoid(relevantSplats.index({ torch::indexing::Slice(), 2 })).to(device);
			auto alpha = torch::sigmoid(relevantSplats.index({ torch::indexing::Slice(), 3 })).to(device);
			auto color = torch::sigmoid(relevantSplats.index({ torch::indexing::Slice(), torch::indexing::Slice(4, 7) })).to(device);
			auto mean = torch::tanh(relevantSplats.index({ torch::indexing::Slice(), torch::indexing::Slice(7, 9) })).to(device);
			
			//Create and render new splats with the new parameter values
			torch::Tensor splatImage = renderSplatImage(height, width, scaleX, scaleY, theta, color * alpha.view({ relevantSplatsNum, 1 }), mean).to(device);

			//Loss calculation - compare rendered image with target image
			auto loss = computeLoss(splatImage, targetImage);
			std::cout << ", Loss: " << loss.item() << std::endl;

			// Backward pass and optimization step
			//Zero out gradients from previous iteration
			optimizer.zero_grad();
			//Compute gradients
			loss.backward();

			//Zero unnecessary gradients
			if (relevantSplatMask.defined()) {
				splatParams.grad().index_put_({ ~relevantSplatMask }, torch::zeros_like(splatParams.grad().index({ ~relevantSplatMask })));
			}

			if (iter % refineIterations == 0 && iter > 0) {
				// Find indices to remove and update the relevance mask
				
				// Remove splats with too small alphas 
				torch::Tensor removeIndices = (torch::sigmoid(splatParams.index({ torch::indexing::Slice(), 3 })) < alphaThreshold).nonzero().view(-1).to(device);

				if (removeIndices.size(0) > 0) {
					std::cout << "Removed " << removeIndices.size(0) << " points." << std::endl;
					relevantSplatMask.index_put_({ removeIndices }, false);
				}					
					
				//Remove completely splats from parameter list OR
				//make parameters and gradients equal zero
				splatParams.data().index_put_({ ~relevantSplatMask }, torch::zeros_like(splatParams.index({ ~relevantSplatMask })));
				

				// Calculate the L2 norm of gradients
				torch::Tensor selectedGrad = splatParams.grad().index({ relevantSplatMask }).to(device);
				torch::Tensor gradientNorms = torch::norm(selectedGrad.index({ torch::indexing::Slice(), torch::indexing::Slice(7, 9) }), 2, 1).to(device);
				torch::Tensor selectedData = torch::sigmoid(splatParams.data().index({ relevantSplatMask })).to(device);
				torch::Tensor gaussianNorms = torch::norm(selectedData.index({ torch::indexing::Slice(), torch::indexing::Slice(0, 2) }), 2, 1).to(device);
				
				// Sort gradients and Gaussian norms
				torch::Tensor sortedGrads;
				torch::Tensor sortedGradsIndices;
				std::tie(sortedGrads, sortedGradsIndices) = torch::sort(gradientNorms, -1, true);
				torch::Tensor sortedGauss;
				torch::Tensor sortedGaussIndices;
				std::tie(sortedGauss, sortedGaussIndices) = torch::sort(gaussianNorms, -1, true);

				// Create masks based on thresholds
				torch::Tensor largeGradientMask = (sortedGrads > gradientThreshold).to(device);
				auto largeGradientIndices = sortedGradsIndices.masked_select(largeGradientMask).to(device);

				torch::Tensor largeGaussMask = (sortedGauss > gaussianThreshold).to(device);
				auto largeGaussIndices = sortedGaussIndices.masked_select(largeGaussMask).to(device);

				// Find those with large gradient and large gaussian values - splats for splitting
				torch::Tensor splitIndicesMask = torch::isin(largeGradientIndices, largeGaussIndices).to(device);
				auto splitIndices = largeGradientIndices.index({ splitIndicesMask }).to(device);

				// Find those with large gradient and small gaussian values - splats for cloning
				auto smallGaussIndices = sortedGaussIndices.index({~largeGaussMask }).to(device);
				torch::Tensor cloneIndicesMask = torch::isin(largeGradientIndices, smallGaussIndices).to(device);
				auto cloneIndices = largeGradientIndices.index({ cloneIndicesMask }).to(device);
				
				// Split splats with large coordinate gradient and large gaussian values and descale their gaussian
				if (splitIndices.size(0) > 0) {					
					int newSplatsBegin = relevantSplatsEnd;
					int newSplatsEnd = relevantSplatsEnd + splitIndices.size(0);
					if(newSplatsEnd > relevantSplatMask.size(0))
					{
						newSplatsEnd = relevantSplatMask.size(0);//clamp to max size of new indices
						splitIndices = splitIndices.index({ torch::indexing::Slice(0, (newSplatsEnd - newSplatsBegin))});
					}

					if((newSplatsEnd - newSplatsBegin) > 0)
					{
						std::cout << "Split " << (newSplatsEnd - newSplatsBegin) << " points." << std::endl;
						relevantSplatMask.index_put_({ torch::indexing::Slice(newSplatsBegin, newSplatsEnd) }, true);
						//Append new splats
						splatParams.data().index_put_({ torch::indexing::Slice(newSplatsBegin, newSplatsEnd), torch::indexing::Slice() }, splatParams.data().index({ splitIndices, torch::indexing::Slice() }));
						
						//Scale down new splats
						splatParams.data().index_put_({ torch::indexing::Slice(newSplatsBegin, newSplatsEnd), torch::indexing::Slice(0, 2) }, splatParams.data().index({ splitIndices, torch::indexing::Slice(0, 2) }) / 1.6);
						//Scale down old original splats
						splatParams.data().index_put_({ splitIndices, torch::indexing::Slice(0, 2) }, splatParams.index({ splitIndices, torch::indexing::Slice(0, 2) }) / 1.6);
						relevantSplatsEnd += splitIndices.size(0);
					}
				}

				// Clone splats with large coordinate gradient and small Gaussian values
				if (cloneIndices.size(0) > 0) {					
					int newSplatsBegin = relevantSplatsEnd;
					int newSplatsEnd = relevantSplatsEnd + cloneIndices.size(0);
					if (newSplatsEnd > relevantSplatMask.size(0)) {
						newSplatsEnd = relevantSplatMask.size(0);//clamp to max size of new indices
						cloneIndices = cloneIndices.index({ torch::indexing::Slice(0, (newSplatsEnd - newSplatsBegin))});
					}

					if ((newSplatsEnd - newSplatsBegin) > 0)
					{
						std::cout << "Cloned " << (newSplatsEnd - newSplatsBegin) << " points." << std::endl;
						relevantSplatMask.index_put_({ torch::indexing::Slice(newSplatsBegin, newSplatsEnd) }, true);
						// Append new splats
						splatParams.data().index_put_({ torch::indexing::Slice(newSplatsBegin, newSplatsEnd), torch::indexing::Slice() }, splatParams.data().index({ cloneIndices, torch::indexing::Slice() }));
						// Move new splats in the direction of the positional gradient
						splatParams.data().index_put_({ torch::indexing::Slice(newSplatsBegin, newSplatsEnd), torch::indexing::Slice(7, 9) },
							splatParams.data().index({ cloneIndices, torch::indexing::Slice(7, 9) }) +
							splatParams.grad().index({ cloneIndices, torch::indexing::Slice(7, 9) }) * 0.1);
						relevantSplatsEnd += cloneIndices.size(0);
					}
				}
			}
			
			//Update parameters
			optimizer.step();

			std::cout << "Iteration " << iter + 1 << " splats: " << relevantSplatsNum << ", loss: " << loss.item() << std::endl;
			
			if (iter % storeIterations == 0)
			{				
				std::cout << "tempShape: " << splatImage.sizes() << std::endl;
				// Reshape needeed to transform data in a linear array of BGR values
				splatImage = splatImage.reshape({width*height*3});
				cv::Mat image_mat(width, height, CV_32FC3, splatImage.data_ptr<float>());
				// Scale the values to the range [0, 255] if needed
				cv::normalize(image_mat, image_mat, 0, 255, cv::NORM_MINMAX);
				std::ostringstream filename;

				//Save output file
				filename <<  "Iter_" << iter + 1 << "_" << iterations << "_loss_" << loss.item() << ".png";
				std::string flname = filename.str();
				cv::String cvstr = flname;
				cv::imwrite(cvstr, image_mat);
			}			
		}
	}
private:
	// Define a function to compute the loss between the generated and target images
	torch::Tensor computeLoss(const torch::Tensor& pred, const torch::Tensor& target) {
		torch::nn::L1Loss l1loss = torch::nn::L1Loss();
		return l1loss(pred, target);
	}		

	torch::Tensor renderSplatImage(int64_t height, int64_t width, const torch::Tensor &_scaleX, const torch::Tensor &_scaleY, const torch::Tensor &_theta,
		const torch::Tensor &colors, const torch::Tensor& mean /*, torch::Tensor alphas*/) {		
		
		namespace F = torch::nn::functional;
		// Get the number of splats
		int64_t numSplats = colors.size(0);
		
		// Reshape parameters to get the same batch dimensions
		torch::Tensor scaleX = _scaleX.reshape({ numSplats, 1, 1 }).to(device);
		torch::Tensor scaleY = _scaleY.reshape({ numSplats, 1, 1 }).to(device);
		torch::Tensor theta = _theta.reshape({ numSplats, 1, 1 }).to(device);

		// Construct rotation matrix:
		torch::Tensor cosTheta = torch::cos(theta * M_PI * 2.0).to(device);
		torch::Tensor sinTheta = torch::sin(theta * M_PI * 2.0).to(device);
		torch::Tensor R = torch::stack({ torch::stack({cosTheta, -sinTheta}, -1),
											torch::stack({sinTheta, cosTheta}, -1) }, -2).to(device);

		// Construct scale matrix:
		torch::Tensor S = torch::stack({ torch::stack({scaleX, torch::zeros({numSplats, 1, 1})}, -1),
											  torch::stack({ torch::zeros({numSplats, 1, 1}), scaleY}, -1) }, -2).to(device);
		// Get transpose matrices
		torch::Tensor RT = (R.transpose(3, 4)).to(device);
		torch::Tensor ST = (S.transpose(3, 4)).to(device);
		
		// Construct covariance matrix:
		torch::Tensor cov = torch::matmul(R, torch::matmul(S, torch::matmul(ST, RT))).to(device);
		
		// Covariance matrix determinant should be positive
		torch::Tensor det = torch::det(cov).to(device);
		if (torch::any(det <= 0).item<bool>()) {
			throw std::invalid_argument("Negative determinant");
		}
		det = det.view({ numSplats, 1, 1 });

		// Inverse of covariance matrix
		torch::Tensor invCov = torch::inverse(cov).to(device);
		
		// Create splatSize evenly spaced values between 0 and 1 for the Gaussian distribution
		torch::Tensor linspace = torch::linspace(-5.0, 5.0, splatSize, torch::kFloat32).reshape({ 1, splatSize }).to(device);
		//Prepare the point x and y values ready for distance calculation		
		torch::Tensor p = torch::stack({ linspace.unsqueeze(-1).expand({-1, -1, splatSize}),
										  linspace.unsqueeze(1).expand({-1, splatSize, -1}) }, -1).to(device);
		
		// Find distance between each point and the distribution:
		// Find a batch-wise computation of the distance between each point in the grid and the 
		// corresponding covariance matrix for each splat in the batch
		//b...i corresponds to transpose point coords, b...ij corresponds to the 2x2 matrices in invCovariance, b...j is p
		torch::Tensor sum = torch::einsum("b...i,b...ij,b...j->b...", { p, invCov, p }).to(device);
		
		//Each value in the pdf tensor represents the weight or intensity of the Gaussian distribution at the corresponding position in the 2D space
		torch::Tensor pdf = (torch::exp(-0.5*sum) / (2.0 * M_PI * torch::sqrt(det))).to(device);

		// Normalize pdf finding max values along last 2 dimensions
		torch::Tensor maxValues = torch::amax(pdf, { -1, -2 }, true).to(device);
		pdf = pdf / maxValues;

		// Include dimensions for the color channels
		torch::Tensor pdfChannels = pdf.unsqueeze(0).repeat({1, 1, 3, 1}).view({numSplats, 3, splatSize, splatSize}).to(device);
		
		// Add offsets so that each splat is placed within the same dimensions as the image
		int64_t offsetH = height - splatSize;
		int64_t offsetW = width - splatSize;
		int64_t offsetHalfH = offsetH * 0.5;
		int64_t offsetHalfW = offsetW * 0.5;
		
		// Fill the tensor with zeros to ensure the image size is the same:
		pdfChannels = torch::constant_pad_nd(pdfChannels, { offsetHalfW, offsetHalfW + offsetW % 2, offsetHalfH, offsetHalfH + offsetH % 2 }, 0).to(device);

		// Construct translation matrix based on splat positions
		torch::Tensor T = torch::zeros({ numSplats, 2, 3 }, torch::dtype(torch::kFloat32)).to(device);
		T.index_put_({ torch::indexing::Slice(), 0, 0 }, 1.0);
		T.index_put_({ torch::indexing::Slice(), 1, 1 }, 1.0);
		T.index_put_({ torch::indexing::Slice(), torch::indexing::Slice(), 2 }, mean);

		// Apply grid sampling to the padded and translated tensor using the generated grid. 
		// This operation simulates the effect of translating the Gaussian splats in the image.
		torch::Tensor grid = torch::nn::functional::affine_grid(T, pdfChannels.sizes(), true).to(device);
		torch::Tensor pdfChannelsT = torch::nn::functional::grid_sample(pdfChannels, grid,
			F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true)).to(device);

		torch::Tensor colorValues = colors.unsqueeze(-1).unsqueeze(-1).to(device);
		// Each color value is multiplied with the corresponding elements in the splat array.
		torch::Tensor batchImage = (colorValues * pdfChannelsT).to(device);

		// Sum color values combining contributions from all splats in the batch
		torch::Tensor renderedImage = batchImage.sum(0).to(device);
		
		//Convert to the same shape and range as target image
		renderedImage = renderedImage.clamp_(0, 1).permute({ 1, 2, 0 });

		return renderedImage;
	}
};



int main() {

	torch::DeviceType deviceType;
	if (torch::cuda::is_available()) {
		std::cout << "CUDA is available.\n";
		deviceType = torch::kCUDA;//torch::kCPU;
	}
	else {
		std::cout << "CUDA is NOT available.\n";
		deviceType = torch::kCPU;
	}

	torch::Device device(deviceType);
	
	// Set the learning rate for the Adam optimizer	
	const double learningRate = 0.01;

	cv::Mat image = cv::imread("ara_64.png", cv::IMREAD_COLOR);
	//cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
	torch::Tensor tensorTarget = torch::from_blob(image.data, { image.rows, image.cols, image.channels() }, torch::kByte).to(device);
	tensorTarget = tensorTarget.to(torch::kFloat) / 255.0;
	
	if (image.empty()) {
		std::cerr << "Failed to load image " << std::endl;
		return 0;
	}	

	// Set the width and height of the image
	int64_t height = tensorTarget.size(0);
	int64_t width = tensorTarget.size(1);
	int64_t numPixels = height*width;
	// Set total number of splats = 10%
	const int numSplats = numPixels / 10 + 240;//650;
	// Initial splats number
	const int numInitSplats = numPixels / 30 + 60;//200;

	// Initialize training data
	
	// Set scale and rotation for the splat shape
	torch::Tensor scales = torch::rand({ numSplats, 2 }, torch::kFloat).to(device);
	torch::Tensor theta = torch::rand({ numSplats, 1 }, torch::kFloat).to(device);
	// Set aplha channel to 1:
	torch::Tensor alphas = torch::ones({ numSplats, 1 }, torch::kFloat).to(device);

	// Generate random x-coordinates
	torch::Tensor randomXCoords = torch::randint(0, width, { numSplats }, torch::kLong).to(device);
	// Generate random y-coordinates
	torch::Tensor randomYCoords = torch::randint(0, height, { numSplats }, torch::kLong).to(device);
	// Concatenate x and y coordinates along the last dimension
	torch::Tensor coords = torch::cat({ randomXCoords.unsqueeze(1), randomYCoords.unsqueeze(1) }, 1).to(device);

	// Get the pixel color per coordinate from the target image
	torch::Tensor colors = torch::zeros({ coords.size(0), 3 }, torch::kFloat).to(device);
	
	for (int i = 0; i < coords.size(0); ++i) {
		int x = static_cast<int>(coords[i][0].item<int>());
		int y = static_cast<int>(coords[i][1].item<int>());

		colors[i][0] = tensorTarget[y][x][0];
		colors[i][1] = tensorTarget[y][x][1];
		colors[i][2] = tensorTarget[y][x][2];
	}	

	// Get position mean
	torch::Tensor mean = (coords / torch::tensor({ width, height }, torch::kFloat)).to(torch::kFloat).to(device);
	mean = torch::atanh(mean * 2.0 - 1.0);
	
	// Prepare parameters tensor
	torch::Tensor W = torch::cat({ scales, theta, alphas, colors, mean }, 1);

	// Create an instance of the SplatRenderer
	SplatRenderer renderer(width, height, W, device.type());
	
	// Create an instance of the Adam optimizer
	torch::optim::Adam optimizer(renderer.parameters(), torch::optim::AdamOptions(learningRate));

	// Optimize the splat parameters to match the target image
	renderer.optimize(tensorTarget, optimizer, numSplats, numInitSplats);

	return 0;	
}
