def detect(frame):
    vals = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    img = resize(frame, (64,64,1))
    img = np.expand_dims(img,axis=0)
    if(np.max(img) > 1):
      img = img / 255.0
    prediction = model.predict(img)
    pred = vals[np.argmax(prediction)]
    print(pred)
    return pred

@app.route('/video_feed')
def video():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)