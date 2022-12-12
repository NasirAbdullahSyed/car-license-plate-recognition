import cv2
import math
from object_detection.Preprocess import *

def detectCharsInPlates(listOfPossiblePlates):
    if len(listOfPossiblePlates) == 0:          
        return listOfPossiblePlates
    for possiblePlate in listOfPossiblePlates:
        possiblePlate.imgGrayscale, possiblePlate.imgThresh = preprocess(possiblePlate.imgPlate)
        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx = 1.6, fy = 1.6)
        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgThresh)
        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)
        possiblePlate.strChars = 0
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            if len(listOfListsOfMatchingCharsInPlate[i]) > possiblePlate.strChars:
                possiblePlate.strChars = len(listOfListsOfMatchingCharsInPlate[i])
    return listOfPossiblePlates

def findPossibleCharsInPlate(imgThresh):
    listOfPossibleChars = []
    contours = []
    imgThreshCopy = imgThresh.copy()
    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        possibleChar = CharProbability(contour)
        if checkIfPossibleChar(possibleChar):              # if contour is a possible char, note this does not compare to other chars (yet) . . .
            listOfPossibleChars.append(possibleChar)       # add to list of possible chars
    return listOfPossibleChars


def checkIfPossibleChar(possibleChar):
    if (possibleChar.intBoundingRectArea > 80 and possibleChar.intBoundingRectWidth > 2 and possibleChar.intBoundingRectHeight > 8 and
        0.25 < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < 1.0):
        return True
    else:
        return False


def findListOfListsOfMatchingChars(listOfPossibleChars):
    # with this function, we start off with all the possible chars in one big list
    # the purpose of this function is to re-arrange the one big list of chars into a list of lists of matching chars,
    # note that chars that are not found to be in a group of matches do not need to be considered further
    # this will be the return value
    listOfListsOfMatchingChars = []                  
    # for each possible char in the one big list of chars
    for possibleChar in listOfPossibleChars:                   
        # find all chars in the big list that match the current char
        listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars) 
        # also add the current char to current possible list of matching chars
        listOfMatchingChars.append(possibleChar)
        # if current possible list of matching chars is not long enough to constitute a possible plate
        # jump back to the top of the for loop and try again with next char, note that it's not necessary
        # to save the list in any way since it did not have enough chars to be a possible plate
        if len(listOfMatchingChars) < 3:   
            continue                                                                                                 
        # if we get here, the current list passed test as a "group" or "cluster" of matching chars
        listOfListsOfMatchingChars.append(listOfMatchingChars)      # so add to our list of lists of matching chars
        listOfPossibleCharsWithCurrentMatchesRemoved = []
        # remove the current list of matching chars from the big list so we don't use those same chars twice,
        # make sure to make a new big list for this since we don't want to change the original big list
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))
        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)      # recursive call
        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:        # for each list of matching chars found by recursive call
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)             # add to our original list of lists of matching chars

        # End the recursion
        break
    return listOfListsOfMatchingChars


def findListOfMatchingChars(possibleChar, listOfChars):
    # the purpose of this function is, given a possible char and a big list of possible chars,
    # find all chars in the big list that are a match for the single possible char, and return those matching chars as a list
    listOfMatchingChars = []   # this will be the return value
    # if the char we attempting to find matches for is the exact same char as the char in the big list we are currently checking
    # then we should not include it in the list of matches b/c that would end up double including the current char
    # so do not add to list of matches and jump back to top of for loop
    for possibleMatchingChar in listOfChars:
        if possibleMatchingChar == possibleChar:       
            continue                               
        # compute stuff to see if chars are a match
        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)
        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)
        fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)
        fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)
        # check if chars match
        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * 5.0) and fltAngleBetweenChars < 12.0 and
            fltChangeInArea < 0.5 and fltChangeInWidth < 0.8 and fltChangeInHeight < 0.2):
            listOfMatchingChars.append(possibleMatchingChar)

    return listOfMatchingChars

# use Pythagorean theorem to calculate distance between two chars
def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)
    return math.sqrt((intX ** 2) + (intY ** 2))


def angleBetweenChars(firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))
    if fltAdj != 0.0:  # check to make sure we do not divide by zero if the center X positions are equal, float division by zero will cause a crash in Python
        fltAngleInRad = math.atan(fltOpp / fltAdj)      # if adjacent is not zero, calculate angle
    else:
        fltAngleInRad = 1.5708                          # if adjacent is zero, use this as the angle, this is to be consistent with the C++ version of this program
    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)       # calculate angle in degrees
    return fltAngleInDeg