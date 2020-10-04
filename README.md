## What is this:

This is a Sudoku solver I built using OpenCV to read the grid or table, and then 
uses your choice of Pytesseract or a Pytorch CNN model to read the digits in each grid cell.



## How to use it:

The file *main.py* can be run from a terminal like command prompt, and is built and tested with Python >3.6.

The arguments that can be entered are:
- **--filepath**: *(Required)* String which indicates the path of the Sudoku image file.  

- **--method**: *(Optional)* String which controls if user wants to use Pytesseract (recommended) or Pytorch model (if available).

- **--replace**: *(Optional)* Controls if user wants to change anything misread as a non-digit into a zero digit. 

- **--print-intermediate**: *(Optional)* Integer which controls if the program shold print intermediate images to reveal what 
        it is finding. If 1 is entered, three figures are printed to the directory of *main.py*; if number is larger, then
        additionally each cell of the Sudoku grid is printed with i and j coordinates.

- **--tesseract-path**: *(Possibly required if using Pytesseract option)* String which indicates where the EXE file for 
            Tesseract is located on local machine.
        The default is *'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'*. If Tesseract already in System PATH variable, then enter an empty string ('').
- **--cnn-model**: *(Possibly required if using CNN option)* String which indicates where the Pytorch CNN *state_dict* file is located on local machine.

Example usage:

```commandline
python main.py --filepath ./sudoku_example.png --replace --print-intermediate 1
```
  

## How does it work:

The solver operates as follows:

    1. An image is read from a file path using an OpenCV method, to do more processing on it.
    
    2. The image is turned into a black and white image using thresholding and inverted.
    
    3. A kernel for finding vertical lines is applied to the image, producing something like the below:

![Vertical Lines](/images/sudoku6_vertical_lines.png)

    4. A kernel for finding vertical lines is applied to the image, producing something like the below:

![Horizontl Lines](/images/sudoku6_horizontal_lines.png)

    5. The two vertical and horizontal images are combines to form the grid space without the characters,
        is then its colors are inverted back.

![Grid](/images/sudoku6_grid_lines.png)

    6. Using this grid, the contours and bounding box functions in OpenCV are applied to find all the 
        individuals cells. These cells are found starting left to right, and are ordered based on their areas. 
    
    7. The boxes are then filtered based on a heursitic that a cell is acknolwedged as desirable if its height and width 
        are not much grater nor smaller than the average width and height of all the boxes. This precents from including
        the entire grid's bounding box for example.
    
    8. After filtering for unwanted cells, we arrange in them an array of arrays to indicate which rows and columns they
        belong. This is done using two steps: first gathering them together at the right level, then more accurately arranging the 
        contens of the second-level arrays using the center point coordinates of each box.
    
    9. With the arranged cells ready, we iterate through them to capture each cell's portion of the overall image, and 
        feed that into either Pytesseract or the Pytorch model. Each cell image output is stored until we have an array 
        representation of the original image.
    
    10. Ths Sudoku array is fed into a function called *solver()* which, with a helper function called *possible()*, 
        attempts to solve the Sudoku puzzle using recursion and backtracking. The output is printed on-screen and the 
        option to attempt to find and display another solution is presented in a prompt.

## References

This solver builds on the work done in this very cool [article](https://towardsdatascience.com/a-table-detection-cell-recognition-and-text-extraction-algorithm-to-convert-tables-to-excel-files-902edcf289ec),
and on the Sudoku solver implemented in this [video](https://www.youtube.com/watch?v=G_UYXzGuqvM) by Computerphile.

The Pytorch model is fixed and is the same structure as that of the Pytorch sample for the MNIST data. See below:

```python
# if CNN is opted for, this is executed internally. Therefore the state_dic file must work with the model below:
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

state_dict = torch.load(args.cnn_model)
model = Net()
model.load_state_dict(state_dict)  # load state_dict into model
```
