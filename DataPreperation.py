
from tqdm import tqdm
import pandas as pd
import shutil
import os

#Utility Functions

def extract_tag(video_path):
    return video_path.split("/")[0]

def separate_video_name(video_name):
    return video_name.split("/")[1]

def rectify_video_name(video_name):
    return video_name.split(" ")[0]

def move_videos(df, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for i in tqdm(range(df.shape[0])):
        videoFile = df['video_name'][i].split("/")[-1]
        videoPath = os.path.join("data", videoFile)
        
        shutil.copy2(videoPath, output_dir)
    print()
    print(f"Total videos: {len(os.listdir(output_dir))}")
        


    


#Open text files that contain training videos
trainingFile = open("/Users/andresangel/Desktop/ucfTrainTestList/trainlist01.txt", "r")
temp = trainingFile.read()
videos = temp.split('\n')

#Create train dataframe containing video names
train = pd.DataFrame()
train['video_name'] = videos
train = train[:-1]


#Open the .txt file containing names of test videos
with open("/Users/andresangel/Desktop/ucfTrainTestList/testlist01.txt", "r") as file:
    temp = file.read()
videos = temp.split("\n")

#Create test dataframe containing video names
test = pd.DataFrame()
test["video_name"] = videos
test = test[:-1]



#Dataframe Preparation
train["tag"] = train['video_name'].apply(extract_tag)
train['video_name'] = train['video_name'].apply(separate_video_name)

train["video_name"] = train["video_name"].apply(rectify_video_name)
print('Break One')
print(train.head())

print('Break Two')
test['tag'] = test['video_name'].apply(extract_tag)
test["video_name"] = test['video_name'].apply(separate_video_name)
print(test.head())



#Filtering Top-n Actions

Num = 10
topNActs = train["tag"].value_counts().nlargest(Num).reset_index()["tag"].tolist()
train_new = train[train["tag"].isin(topNActs)]
test_new = test[test["tag"].isin(topNActs)]
print(train_new.shape, test_new.shape)

train_new = train_new.reset_index(drop=True)
test_new = test_new.reset_index(drop=True)

#Move Top-n Action Videos to CSV!


#move_videos(train_new,"train")
#move_videos(test_new,"test")


train_new.to_csv("train.csv", index=False)
test_new.to_csv("test.csv", index=False)
