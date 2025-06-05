from sklearn.metrics.pairwise import cosine_similarity
import joblib
from flask import Flask, request,make_response
import joblib
from flask_cors import CORS
from operator import itemgetter
from dotenv import load_dotenv
import os




load_dotenv()
app = Flask(__name__)
frontend_url = os.getenv('FRONTEND_URL')
CORS(app,origins=[frontend_url])


cv = joblib.load("models/vectorizer.pkl")
vectorMatrix = joblib.load("models/vector_matrix.pkl")
df = joblib.load("models/dataFrame.pkl")


def recommend_by_preferences(ingredients: str, cuisine: str, course: str, diet: str, max_time: str):
    # Combine user input into the same format
    user_input = ' '.join([
        ingredients.lower(),
        cuisine.lower(),
        course.lower(),
        diet.lower(),
        max_time
    ])

    user_vector = cv.transform([user_input])
    similarity_scores = cosine_similarity(user_vector, vectorMatrix).flatten()

    # Get top 5 recommendations
    top_indices = similarity_scores.argsort()[-5:][::-1]
    recommendations = []
    for idx in top_indices:
        recommendations.append(df.iloc[idx].to_dict(), idx)
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

if __name__ == '__main__':
    app.run(debug=True)




