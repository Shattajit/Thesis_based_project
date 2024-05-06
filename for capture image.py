from flask import send_from_directory

@app.route('/captured_images')
def captured_images():
    image_names = os.listdir("collected_faces")
    return render_template('captured_images.html', image_names=image_names)

@app.route('/collected_faces/<path:filename>')
def collected_faces(filename):
    return send_from_directory("collected_faces", filename)
