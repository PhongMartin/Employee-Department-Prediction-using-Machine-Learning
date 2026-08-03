import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import  Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile

def filter_location(location):
    if location[-4:-2] == ", " and location[-2:].isupper():
        return location[-2:]
    else:
        return location


# 1. Đọc dữ liệu
data = pd.read_excel('data/final_project (1).ods', dtype=str)

# 2. Thống kê dữ liệu
# print(data.info())
# RangeIndex: 8074 entries, 0 to 8073
# Data columns (total 6 columns):
#  #   Column        Non-Null Count  Dtype
# ---  ------        --------------  -----
#  0   title         8074 non-null   object
#  1   location      8074 non-null   object
#  2   description   8073 non-null   object
#  3   function      8074 non-null   object
#  4   industry      8074 non-null   object
#  5   career_level  8074 non-null   object
# Sau khi thống kê thì nhận thấy cột description bị thiếu dữ liệu nên loại bỏ dữ liệu này
data = data.dropna(axis = 0)
data["location"] = data["location"].apply(filter_location)
# 3. Chia dữ liệu
# print(data["career_level"].value_counts())
# career_level
# senior_specialist_or_project_manager      4337
# manager_team_leader                       2672
# bereichsleiter                             960
# director_business_unit_leader               70
# specialist                                  30
# managing_director_small_medium_company       4
# Ta nhận thấy đây là dữ liệu mất cân bằng
target = "career_level"
x = data.drop(target, axis = 1)
y = data[target]
# print(y.value_counts())
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size = 0.2,
    random_state = 1101,
    stratify = y,
    # Cái này sẽ làm cho việc chia dữ liệu phân đều hơn
)
# print("-----------------------")
# print(y_train.value_counts())

# 4. Tiền xử lý dữ liệu
# ---------------------------------------------------------------------------
# 4.1 Cột title
# bởi vì đoạn văn ngắn nên dùng unigram
# vectorizer = TfidfVectorizer(stop_words='english')
# result = vectorizer.fit_transform(x_train["title"])
# print(vectorizer.vocabulary_)
# print(len(vectorizer.vocabulary_))
# print(result.shape)
# (6458, 2966)
# ----------------------------------------------------------------------------
# 4.2 Cột location
# encoder = OneHotEncoder()
# result = encoder.fit_transform(x_train[["location"]])
# print(result.shape)
# (6458, 95)
# ----------------------------------------------------------------------------
# 4.3 Cột description
# vectorizer = TfidfVectorizer(stop_words='english', ngram_range = (1,2), min_df = 0.01, max_df = 0.99)
# result = vectorizer.fit_transform(x_train["description"])
# print(vectorizer.vocabulary_)
# print(len(vectorizer.vocabulary_))
# print(result.shape)
# (6458, 4340)
# ----------------------------------------------------------------------------
# 4.4 Cột function
# encoder = OneHotEncoder()
# result = encoder.fit_transform(x_train[["function"]])
# print(result.shape)
# (6458, 19)
# ----------------------------------------------------------------------------
# 4.5 Cột industry
# encoder = OneHotEncoder()
# result = encoder.fit_transform(x_train[["industry"]])
# print(result.shape)
# (6458, 347)
# ----------------------------------------------------------------------------
# Tổng hợp lại
preprocessor = ColumnTransformer(transformers=[
    ("title", TfidfVectorizer(stop_words= "english"), "title"),
    ("location", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    # handle_unknown dùng để giải quyết các dữ liệu không có mặt trong quá trình trainning và đưa nó về vector 0
    ("description", TfidfVectorizer(stop_words="english", ngram_range= (1,2), min_df=0.01, max_df= 0.99), "description"),
    ("function", OneHotEncoder(handle_unknown="ignore"), ["function"]),
    ("industry", TfidfVectorizer(stop_words= "english"), "industry"),
])

model = Pipeline([
    ("preprocessor", preprocessor),#6000 feature
    ("selector", SelectPercentile(chi2, percentile=8)),
    ("classifier", RandomForestClassifier(random_state=1101)),
])

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
#                                       precision    recall  f1-score   support

#                         bereichsleiter       0.76      0.13      0.22       192
#          director_business_unit_leader       1.00      0.29      0.44        14
#                    manager_team_leader       0.63      0.75      0.68       534
# managing_director_small_medium_company       0.00      0.00      0.00         1
#   senior_specialist_or_project_manager       0.84      0.90      0.87       868
#                             specialist       0.00      0.00      0.00         6
#
#                               accuracy                           0.75      1615
#                              macro avg       0.54      0.35      0.37      1615
#                           weighted avg       0.76      0.75      0.72      1615


#          director_business_unit_leader       1.00      0.21      0.35        14
#  Tại sao cái này precision lại cao như vây?

# 6. Triển khai mô hình: Có 2 hướng
# 1. Performance || 2. Tối ưu tốc độ xử lý