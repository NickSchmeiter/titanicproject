# This is a sample Python script
import pandas as pd
from pyexpat import features
from sklearn.ensemble import RandomForestClassifier
# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Strg+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_data = pd.read_csv(r'C:\Users\nicks\Desktop\python\Kaggle\Titanic Project\train.csv')
    print(train_data)
    test_data = pd.read_csv(r'C:\Users\nicks\Desktop\python\Kaggle\Titanic Project\test.csv')
    print(test_data)
    dfwoman=pd.read_csv(r'C:\Users\nicks\Desktop\python\Kaggle\Titanic Project\gender_submission.csv')

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    y = train_data["Survived"]

    features = ["Pclass", "Sex", "SibSp", "Parch"]
    X = pd.get_dummies(train_data[features])
    X_test = pd.get_dummies(test_data[features])
    X.fillna(X.mean())
    model = SVC()
    model.fit(X, y)
    predictions = model.predict(X_test)

    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv('submission.csv', index=False)
    output.to_csv(r'C:\Users\nicks\Desktop\python\Kaggle\Titanic Project\submission.csv', index=False)
    print("Your submission was successfully saved!")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
