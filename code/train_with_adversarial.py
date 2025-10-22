import sys
sys.path.insert(0, '.')

# Modify student_code to use adversarial model
import student_code
student_code.default_cnn_model = student_code.SimpleNetAdversarial

# Import and run main
from main import *

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
