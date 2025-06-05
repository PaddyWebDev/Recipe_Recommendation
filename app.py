from sklearn.metrics.pairwise import cosine_similarity
import joblib
from flask import Flask, request,make_response
import joblib
from flask_cors import CORS
from operator import itemgetter
from openpyxl import load_workbook


app = Flask(__name__)
CORS(app)


cv = joblib.load("models/vectorizer.pkl")
vectorMatrix = joblib.load("models/vector_matrix.pkl")
df = joblib.load("models/dataFrame.pkl")


def recommend_by_preferences(ingredients: str, cuisine: str, course: str, diet: str, max_time: str):
    user_input = ' '.join([
        ingredients.lower(),
        cuisine.lower(),
        course.lower(),
        diet.lower(),
        max_time
    ])

    user_vector = cv.transform([user_input])
    similarity_scores = cosine_similarity(user_vector, vectorMatrix).flatten()

    # Get top 5 similar recipe indices
    top_indices = similarity_scores.argsort()[-5:][::-1]

    recommendations = []
    for idx in top_indices:
        recipe = {
            "idx": int(idx),  # Cast to int for safe JSON serialization
            "title": df.iloc[idx].to_dict()["TranslatedRecipeName"]
        }
        recommendations.append(recipe)

    return recommendations



@app.route('/', methods=["GET"])
def home():
  return make_response("Backend for Recipe Recommendation System",200)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if(not data):
            return make_response({
                "Error" : {
                "code": 400,
                "message": "Bad request"
                }
            }, 400)
        
        ingredients, cuisine, course, diet, cooking_time = itemgetter('ingredients', 'cuisine', 'course', 'diet', 'cooking_time')(data)

        recommendation = recommend_by_preferences(ingredients, cuisine, course, diet, cooking_time)
        
        return make_response({
        "Success": {
        "code": 200,
        "prediction": recommendation
    }}, 200)
    except Exception as e:
        return make_response({"Error": {"code": 500, "message": str(e)}}, 500)


@app.route('/recipe/<recipe_id>', methods=['GET'])
def get_recipe(recipe_id):
    try:
        workbook = load_workbook(filename='./models/data/IndianFoodDatasetXLS.xlsx', data_only=True)
        sheet = workbook.active  # Or workbook['SheetName'] if you know it

        # Read header row (usually first row)
        headers = [cell.value for cell in sheet[1]]

        # Convert recipe_id (zero-based index) to Excel row number (data starts at row 2)
        excel_row_num = int(recipe_id) + 2
        row = sheet[excel_row_num]

        # Extract row values
        row_values = [cell.value for cell in row]

        # Create dictionary by zipping headers with row values
        row_dict = dict(zip(headers, row_values))

        return make_response({
            "Success": {
                "code": 200,
                "recipe": row_dict
            }
        })
    except Exception as e:
        return make_response({"Error": {"code": 500, "message": str(e)}}, 500)
    
if __name__ == '__main__':
    app.run(debug=True)




