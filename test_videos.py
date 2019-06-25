'''
Alexandros I. Metsai
alexmetsai@gmail.com
MIT License

Test the accuracy of the contrained convolutional neural 
network based on predictions of video frame images.
A threshold value is used to make the 
decision for the overall classification.
'''

# Set argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-p',
    '--path',
    type=str,
    help="Path of the network's weights")

# Display help if no arguments are given
if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)

# Folder path to weights
args = parser.parse_args()
path = args.path

# Define image and batch size
img_height = 256        # CHanged image size!!!
img_width = 256
batch_size = 64

negative_data_dir = './test/negative'

# Percentage of tampered frames to classify video as fake.
detection_threshold = 0.5

# Load and Compile the model
model = load_model(path)
sgd = SGD(lr=0.001, momentum=0.95, decay=0.0004)
model.compile(
    optimizer=sgd, 
    loss='binary_crossentropy', 
    metrics=['accuracy'])

# Create the Generator
test_data_gen = ImageDataGenerator(preprocessing_function=None,
    rescale=1./255)

# ****************************
# *** Test negative videos ***
# ****************************
video_folders = os.listdir(negative_data_dir)

# Make video-level prediction
correct_guesses = 0

for i in range(len(video_folders)):
    
    # Read the data from directory
