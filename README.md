# HuynhNgocThang_YoloObjectDetectionInBadminton
A small project to experience yolo for object detection + tracking and pytorch for keypoint mapping.

# The base guideline
## This part is for finding document and research about the project

- This project is guided/ based on another project called "Build an AI/ML Tennis Analysis system with YOLO, PyTorch, and Key Point Extraction" by Code In a Jiffy (Youtube) [https://www.youtube.com/watch?v=L23oIHZE14w&t=3331s]
- Our goal is to create another version on Badminton Analysis system with YOLO, PyTorch and Keypoint Extraction

 - First is Object detection: We're going to detect many different kinds of object that is included in a badminton match. This could be the Player, Shuttecock, etc... This will be done by our object detection model YOLO which going to be trained on a specific kind of dataset that has already annotationed.

 - Secondly is Tracking: After we detect our object on a video frame, we then need to track it consistently to tell the different between different objects of the same kind when it's location changed in the next video frame (this apply to a kind of object that appear multiple times on a video frame). This is done by using ByteTracker from supervision library.
 For example: Player A is on one side and player B is on another side, after the model detect that object is a player. It then assign the player A ID's is 1 and player B ID's is 2, so even if the player moved we still know that the player A ID's is 1 or the object class "Player" that have ID = 1 is the player A

# Part 01: Introdution
## Testing out YOLO models from ultralytics
We're going to try out an already trained YOLO models from the ultralytics library
This already trained model can detect "popular" object like person, ball, chair...

The detection is then showed on a result video with bounding box, class name, and the confident 

