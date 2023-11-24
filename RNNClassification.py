import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

from tensorflow_docs.vis import embed
from tensorflow import keras
from imutils import paths
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2



Img_Size = 224
Batch_Size = 64
epochs = 10

max_seq_length = 20
num_features = 2048

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(f"Total videos for training: {len(train_df)}")
print(f"Total videos for testing: {len(test_df)}")

#Prints out random assortment of items with train dataframe
#print(train_df.sample(10))


'''
One of the many challenges of training video classifiers is figuring out a way to 
feed the videos to a network. This blog post discusses five such methods. Since a 
video is an ordered sequence of frames, we could just extract the frames and put them 
in a 3D tensor. But the number of frames may differ from video to video which would 
prevent us from stacking them into batches (unless we use padding). As an alternative, 
we can save video frames at a fixed interval until a maximum frame count is reached.


Objectives
1.Capture the frames of a video
2.Extract frames from the videos until a maximum frame count is reached.
3.In the case, where a video's frame count is lesser than the maximum frame count we will pad the video with zeros.
'''

'''
# The following two methods are taken from this tutorial:
# https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub

Keep in mind that the process for solving issues involving text sequences is the same. 
It is recognized that there aren't many noticeable differences between objects and activities in videos from the UCF101 dataset. 
For the learning job, it could be acceptable to take into account just a few frames as a result. However, there's a chance that 
this method won't work well for other video categorization issues. The VideoCapture() function in OpenCV will be utilized to read frames from videos.
'''

def crop_center_square(frame):
    y,x = frame.shape[0:2]
    min_dim = min(y,x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def load_video(path, max_frames=0, resize=(Img_Size, Img_Size)):
    cap = cv2.VideoCapture(path)
    frames = []
    try: 
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame,resize)
            frame = frame[:,:, [2,1,0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break

    finally:
        cap.release()

    return np.array(frames)

'''
From the collected frames, we may extract useful characteristics 
using a pre-trained network. Several cutting-edge models that have 
already been trained on the ImageNet-1k dataset are available through 
the Keras Applications module. For this, we'll be use the InceptionV3 model

'''

def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(Img_Size,Img_Size,3),
    )

    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((Img_Size,Img_Size,3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs,outputs,name="feature_extractor")

feature_extractor = build_feature_extractor()

'''
The video labels are strings. String values must be transformed 
into a numerical format before being supplied to the model as neural 
networks are incapable of understanding them. In this case, the class 
labels will be encoded as numbers using the StringLookup layer.

'''

label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["tag"])
)

print(label_processor.get_vocabulary())

#Data Processing Utility

def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df['video_name'].values.tolist()
    labels = df['tag'].values
    labels = label_processor(labels[..., None]).numpy()


    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.

    frame_masks = np.zeros(shape=(num_samples, max_seq_length), dtype='bool')
    frame_features = np.zeros(
        shape=(num_samples, max_seq_length, num_features),dtype="float32"
    )

    #For each video

    for idx, path in enumerate(video_paths):
        frames = load_video(os.path.join(root_dir, path))
        
        frames = frames[None,...]
        
        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(shape=(1,max_seq_length,), dtype='bool')
        temp_frame_features = np.zeros(
            shape=(1,max_seq_length,num_features), dtype='float32'
        )

        #Extract features from the frames of the current video
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(max_seq_length,video_length)
            for j in range(length):
                temp_frame_features[i,j,:] = feature_extractor.predict(
                    batch[None,j,:]
                )

            temp_frame_mask[i,:length] = 1 # 1 = not masked, 0 = masked

        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks), labels

print("file exists?", os.path.exists('/Users/andresangel/Desktop/train/v_BoxingPunchingBag_g08_c01.avi'))
train_data, train_labels = prepare_all_videos(train_df, "/Users/andresangel/Desktop/train")
print()
print("Training CV videos done")
test_data, test_labels = prepare_all_videos(test_df, "/Users/andresangel/Desktop/test")

#Will take about 20 minutes to run
print()
print("Testing CV videos done")

print(f"Frame features in train set: {train_data[0].shape}")
print(f"Frame masks in train set: {train_data[1].shape}")

#Utility function for Sequence Model
def get_sequence_model():
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((max_seq_length,num_features))
    mask_input = keras.Input((max_seq_length,),dtype="bool")

    # https://keras.io/api/layers/recurrent_layers/gru/
    # Follow link above for info on hidden layer

    x = keras.layers.GRU(16, return_sequences=True) (
        frame_features_input, mask=mask_input
    )
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation='relu')(x)
    output = keras.layers.Dense(len(class_vocab), activation='softmax')(x)

    rnn_model = keras.Model([frame_features_input,mask_input], output)
    
    rnn_model.compile(
        loss = "sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    return rnn_model


def run_experiment():
    filepath = "/tmp/video_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath,save_weights_only=True, save_best_only=True, verbose = 1
    )

    seq_model = get_sequence_model()
    history = seq_model.fit(
        [train_data[0], train_data[1]],
        train_labels,
        validation_split=0.3,
        epochs=epochs,
        callbacks=[checkpoint],
    )

    seq_model.load_weights(filepath)
    _,accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
    print(f"Test Accuracy: {round(accuracy * 100, 2)}%")

    return history, seq_model


_, sequence_model = run_experiment()
sequence_model.save("my_model.keras")


def prepare_single_video(frame):
    frames = frames[None,...]
    frame_mask = np.zeros(shape=(1,max_seq_length,), dtype="bool")
    frame_features = np.zeros(shape=(1,max_seq_length,num_features), dtype="float32")

    for i,batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(max_seq_length, video_length)
        for j in range(length):
            frame_features[i,j, :] = feature_extractor.predict(batch[None,j,:])

        frame_mask[i,:length] = 1 # 1 = not masked, 0 = masked

    return frame_features, frame_mask


def sequence_prediction(path):
    class_vocab = label_processor.get_vocabulary()

    frames = load_video(os.path.join("/Users/andresangel/Desktop/test", path))
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = sequence_model.predict([frame_features,frame_mask])[0]

    for i in np.argsort(probabilities)[::-1]:
        print(f" {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")

    return frames

def to_gif(images):
    converted_images = images.astype(np.uint8)
    imageio.mimsave("animation.gif", converted_images, duration=100)
    return embed.embed_file("animation.gif")

test_video = np.random.choice(test_df["video_name"].values.tolist())
print(f"Test Video Path: {test_video}")
test_frames = sequence_prediction(test_video)
to_gif(test_frames[:max_seq_length]) 
