from flask import Flask

app = Flask(__name__)

 
 
# Route for seeing a data
@app.route('/generate-ad')
def func():
 
    # Returning an api for showing in  reactjs
    return {
        
        }

if __name__ == '__main__':
    app.run(debug=True)