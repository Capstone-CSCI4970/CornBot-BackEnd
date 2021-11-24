
#Refrence: https://github.com/DTrimarchi10/confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import io
import matplotlib
matplotlib.use('Agg')
def user_acc_viz(users,acc):
    np.random.seed(0)
    plt.rcdefaults()
    fig, ax = plt.subplots()

    y_pos = np.arange(len(users))

    ax.barh(y_pos, acc ,align='center')
    ax.set_yticks(y_pos)#, labels=people)
    ax.set_yticklabels(users)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('UserName')
    ax.set_title('Accuracies based on user')
    plt.savefig("UserNacc.jpg")
    buf = io.BytesIO()
    plt.gcf().savefig(buf, format='png')
    buf.seek(0)
    user_acc = base64.b64encode(buf.read()).decode('ascii')
    return user_acc

def image_misclass_viz(img_names,num):
    np.random.seed(0)
    img_names = [x.split('.')[0][3:] for x in img_names]
    plt.rcdefaults()
    fig, ax = plt.subplots()

    y_pos = np.arange(len(img_names))

    ax.barh(y_pos, num ,align='center')
    ax.set_yticks(y_pos)#, labels=people
    ax.set_yticklabels(img_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Number of Misclassfied')
    ax.set_ylabel('Image Name')
    ax.set_title('Top 5 Image Misclassified Based On Users')
    plt.savefig("ImageMiss.jpg")
    buf = io.BytesIO()
    plt.gcf().savefig(buf, format='png')
    buf.seek(0)
    img_msclas = base64.b64encode(buf.read()).decode('ascii')
    return img_msclas
