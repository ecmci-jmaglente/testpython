from flask import Flask, request, jsonify
from deepface import DeepFace

app = Flask(__name__)


@app.route('/findFace', methods=['POST'])
def analyze_face():
    data = request.json
    base64_image = data.get('image')

    if not base64_image:
        return jsonify({"error": "No image provided"}), 400

    try:
        face_objs = DeepFace.extract_faces(
            img_path=base64_image,
            anti_spoofing=True
        )

        if all(face_obj["is_real"] for face_obj in face_objs):
            dfs = DeepFace.find(
                img_path=base64_image,
                db_path="database",
            )

            # Convert list of DataFrames to a serializable format
            result = []
            for df in dfs:
                result.append(df.to_dict(orient='records'))

            return jsonify({"result": result})
        else:
            return jsonify({"result": "Some faces might be spoofed or fake."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)