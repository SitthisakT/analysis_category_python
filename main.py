from flask import Flask , jsonify , request ,render_template
import twitter_predict_one


app = Flask(__name__)

@app.route('/text', methods=['POST'])
def textContent(): 
   data = request.json
   print(data)
   return twitter_predict_one.predict_twitter_text(data)

# @app.route('/file', methods=['POST']) 
# def fileContent():
#     data = request.json
#     print(data)
#     return twitter_predict_one.predict_twitter_text(data)


if __name__ == '__main__':
   app.run(port=5000, debug=True)