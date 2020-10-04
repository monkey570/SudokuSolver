import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    from PIL import Image
except ImportError:
    import Image

# Taken from Computerphile's YouTube video "Python Sodoku Solver"
def possible(y,x,n,m, grid):
    for i in range(0,m):
        if grid[y][i] == n:
            return False
    for i in range(0,m):
        if grid[i][x] == n:
            return False
    x0 = (x // 3) * 3
    y0 = (y // 3) * 3
    subsquare = int(np.sqrt(m))
    for i in range(0, subsquare):
        for j in range(0, subsquare):
            if grid[y0+i][x0+j] == n:
                return False
    return True

def solver(m, grid):
    m = int(m)
    for y in range(m):
        for x in range(m):
            if grid[y][x] == 0:
                for n in range(1,m+1):
                    if possible(y,x,n,m,grid):
                        grid[y][x] = n
                        # employ recursion
                        solver(m, grid)
                        # employ backtracking in case of failure
                        grid[y][x] = 0
                return
    print(np.array(grid))
    print('\n')
    reply = input("Check for more solutions? (Type N to exit or if solution repeats)")
    if 'n' in reply.lower():
        print('Bye!')
        exit()

def solveSodoku(name):
    parser = argparse.ArgumentParser(description='Sodoku solver')
    parser.add_argument('--filepath', type=str, default='',
                        help='Enter a file path with file name and extension')
    parser.add_argument('--method', type=str, default='tesseract',
                        help='Choose between "tesseract" or "cnn"')
    parser.add_argument('--replace', action='store_true', default=False,
                        help='Choose to replace any digits misread as alphabet with digit zero. Not this is useful '
                             'for misread images but the correct image digit may not be zero.')
    parser.add_argument('--print-intermediate', type=int, default=0,
                        help='Choose 1 to print intermediate outputs for inspection. Choose 2 to also print individual boxes.')
    parser.add_argument('--tesseract-path', type=str, default='C:\\Program Files\\Tesseract-OCR\\tesseract.exe',
                        help='Enter a file path for "tesseract" exe file')
    parser.add_argument('--cnn-model', type=str, default='./mnist_cnn.pt',
                        help='Enter a file path for a cnn pytroch model\'s state dict. The default is '
                    './mnist_cnn.pt from the pytroch sample code which produces and saves the state dict with that name')
    args = parser.parse_args()

    if 'tesseract' in args.method:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_path
    elif 'cnn' in args.method:
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

    else:
        raise Exception('Please choose a method, either "cnn", or "tesseract".')

    # load the image file
    file = args.filepath
    filename = file
    try:
        filename = file.split('/')[-1].split('.')[0]
    except:
        try:
            filename = file.split('\\')[-1].split('.')[0]
        except:
            filename = file


    img = cv2.imread(file, 0)

    # preprocess the image to separate the horizontal and vertical lines, and recombine into empty grid/table
    thresh, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    try:
        img_bin = 255 - img_bin
    except:
        raise Exception('Image file type has produced an error, or check file path and extension are correct.')
    kernel_len = np.round(np.array(img_bin).shape[1]/100).__int__() * 3
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
    if args.print_intermediate > 0:
        cv2.imwrite(filename+'_vertical_lines_.png',vertical_lines)
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=2)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
    if args.print_intermediate > 0:
        cv2.imwrite(filename+'_horizontal_lines.png', horizontal_lines)
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    if args.print_intermediate > 0:
        cv2.imwrite(filename + '_grid_lines.png', img_vh)
    thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bitxor = cv2.bitwise_xor(img, img_vh)
    bitnot = cv2.bitwise_not(bitxor)

    # Detect contours for following box detection and sort all the contours by top to bottom.
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    def sort_contours(cnts, method="left-to-right"):
        reverse = False
        i = 0
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(
            *sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))  # test later
        return (cnts, boundingBoxes)
    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")

    # find mean height and width of the cells
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    mean_height = np.mean(heights)
    widths = [boundingBoxes[i][2] for i in range(len(boundingBoxes))]
    mean_width = np.mean(widths)

    # Create list box to store all boxes in
    box = []
    # Get position (x,y), width and height for every contour and show the contour on image
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w < (mean_width * 2)) and (h < (mean_height * 2)) and (w > (mean_width * 0.5)) and (
                h > (mean_height * 0.5)):
            # image = cv2.rectangle(image_temp, (x, y), (x + w, y + h), (0, 255, 0), 2)
            box.append([x, y, w, h])

    # Creating two lists to define row and column in which cell is located
    row = []
    column = []
    j = 0
    # Sorting the boxes to their respective row and column
    for i in range(len(box)):
        if (i == 0):
            column.append(box[i])
            previous = box[i]
        else:
            if (box[i][1] <= previous[1] + mean_height / 2):
                column.append(box[i])
                previous = box[i]
                if (i == len(box) - 1):
                    row.append(column)
            else:
                row.append(column)
                column = []
                previous = box[i]
                column.append(box[i])

    # calculating maximum number of cells
    countcol = 0
    for i in range(len(row)):
        countcol = len(row[i])
        if countcol > countcol:
            countcol = countcol
    # Retrieving the center of each column
    center = [int(row[i][j][0] + row[i][j][2] / 2) for j in range(len(row[i])) if row[0]]
    center = np.array(center)
    center.sort()

    # arrange the 'boxes' or cells in correct order based on where their centers are within each row
    finalboxes = []
    for i in range(len(row)):
        lis = []
        for k in range(countcol):
            lis.append([])
        for j in range(len(row[i])):
            diff = abs(center - (row[i][j][0] + row[i][j][2] / 4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        finalboxes.append(lis)

    # from every single image-based cell/box the digits are extracted via pytesseract or a cnn and stored in a list
    errflag=False
    outer = []
    for i in range(len(finalboxes)):
        for j in range(len(finalboxes[i])):
            inner = ''
            if (len(finalboxes[i][j]) == 0):
                outer.append(' ')
                # pass
            else:
                for k in range(len(finalboxes[i][j])):
                    y, x, w, h = finalboxes[i][j][k][0], finalboxes[i][j][k][1], finalboxes[i][j][k][2], \
                                 finalboxes[i][j][k][3]
                    finalimg = bitnot.copy()[x:x + h, y:y + w]  # img_vh
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                    border = cv2.copyMakeBorder(finalimg, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
                    resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    dilation = cv2.dilate(resizing, kernel, iterations=1)
                    erosion = cv2.erode(dilation, kernel, iterations=1)
                    if args.print_intermediate > 1:
                        cv2.imwrite(filename + 'box_i' + str(i) + '_j' + str(j) + '.png', erosion)
                    if 'tesseract' in args.method:
                        out = pytesseract.image_to_string(erosion, config='--psm 6').strip()
                        if (len(out) == 0):
                            out = pytesseract.image_to_string(erosion, config='--psm 3').strip()
                            if (len(out) == 0):
                                out = '0'
                        if (not out.isnumeric()) and args.replace:
                            out = '0'
                    elif 'cnn' in args.method:
                        if (erosion.shape[0] != 28) or (erosion.shape[1] != 28):
                            # image = imutils.resize((image.copy()), width=28, height=28, inter = cv2.INTER_CUBIC) #cv2.bitwise_not
                            erosion = cv2.resize(erosion, (28, 28), interpolation=cv2.INTER_CUBIC)
                        img_new = erosion.reshape(1, 28, 28).astype(np.float32)
                        img_new = np.expand_dims(img_new, 1)
                        input_tensor = torch.tensor(img_new)
                        output = model(input_tensor)
                        pred = output.argmax(dim=1, keepdim=True)
                        out = pred.numpy()[0][0]
                    if k > 0:
                        inner = inner + " " + out
                    else:
                        inner = out
                    # inner.append(int(out))
                try:
                    outer.append(int(inner))
                except:
                    outer.append(inner)
                    errflag = True
    arr = np.array(outer)
    arr = arr.reshape(len(row), countcol)

    print('Original problem scanned as: \n')
    print(arr)
    print('\n')
    if errflag:
        raise Exception('At least one digit was read incorrectly as non-digit character(s), '
                        'therefore no resulting solution. Please use better image or try other method.')
    # test grid
    # grid = [[5,3,0,0,7,0,0,0,0],[6,0,0,1,9,5,0,0,0],[0,9,8,0,0,0,0,6,0],[8,0,0,0,6,0,0,0,3],[4,0,0,8,0,3,0,0,1],[7,0,0,0,2,0,0,0,6],[0,6,0,0,0,0,2,8,0],[0,0,0,4,1,9,0,0,5],[0,0,0,0,8,0,0,7,9]]
    else:
        print('Solution: \n')
        solver(len(finalboxes), arr)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    solveSodoku('PyCharm')
