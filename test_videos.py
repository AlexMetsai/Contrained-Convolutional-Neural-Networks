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
