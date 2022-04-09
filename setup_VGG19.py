# import the necessary packages
from keras.applications import vgg19
from keras.applications import imagenet_utils


class ModelVGG19:
	image_size=224
	num_labels=1000
	num_channels=3

	def predict(self,image):
		# classify the image
		Network = vgg19.VGG19
		model = Network(weights="imagenet")
		print("[INFO] classifying image with VGG19")
		preds = model.predict(image)
		P = imagenet_utils.decode_predictions(preds)

		# loop over the predictions and display the rank-5 predictions +
		# probabilities to our terminal
		for (i, (imagenetID, label, prob)) in enumerate(P[0]):
			print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

