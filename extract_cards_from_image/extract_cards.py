import cv2
import argparse

def get_cards_from_image(filename):

    image= cv2.imread(filename)

    #trial and error to obtain the best values
    edges= cv2.Canny(image, 180,280)

    # saves the gray sacle image for debug
    cv2.imwrite('gray_image.jpg', edges) 

    #find all the contours in the image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #sort the contours using the enclosed area
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=False)

    #sorted from the greatest contour, to the smallest
    sorted_contours.reverse()

    #gets the 13 (random guess value) biggest countours
    #TODO: make sure the contours have no overlaps or are inside each other
    num_contours_to_get = 13
    biggest_countours = sorted_contours[:num_contours_to_get]

    count = 0
    img_contour = image.copy()
    for contour in biggest_countours:

        filename = "card_" + str(count) + ".jpg"
        
        #gets the rectangle associated with the contour
        x_top_left,y_top_left,width,height= cv2.boundingRect(contour)

        print("Card coordinate:")        
        print("x = {}, y = {}, width = {}, height = {}".format(x_top_left,y_top_left,width,height))
        
        #crops the cards from the image
        cropped_image = img_contour[y_top_left:y_top_left+height, x_top_left:x_top_left+width, :]
        
        print("Cropped card size width={} height={}".format(cropped_image.shape[0], cropped_image.shape[1]))
        
        #if the width is greater than the height, rotate 90 degs, so all the cards are up
        if cropped_image.shape[0] < cropped_image.shape[1]:
            print("Rotating image {} by 90 degrees".format(filename))
            cropped_image = cv2.rotate(cropped_image, cv2.cv2.ROTATE_90_CLOCKWISE)

        print("\n\n")

        cv2.imwrite(filename, cropped_image)    
        count = count + 1

    #saves an image with the contour in the cards, to debug
    img_rect = image.copy()    
    for contour in biggest_countours:
        
        #gets the rectangle associated with the contour
        x_top_left,y_top_left,width,height = cv2.boundingRect(contour)

        cv2.rectangle(img_rect, (x_top_left,y_top_left), (x_top_left+width,y_top_left+height),  (0,255,0), 3)
        
    cv2.imwrite('with_rect.jpg', img_rect)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Card extractor')
    parser.add_argument('filename', help='original image to extract the cards from')

    args = parser.parse_args()
    
    get_cards_from_image(args.filename)
