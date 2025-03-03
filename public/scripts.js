console.log(faceapi);

const run = async () => {
    // Load webcam stream
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    const videoFeedEl = document.getElementById("video-feed");

    // Load models before starting the video
    await Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromUri('./models'),
        faceapi.nets.faceLandmark68Net.loadFromUri('./models'),
        faceapi.nets.faceRecognitionNet.loadFromUri('./models'),
        faceapi.nets.ageGenderNet.loadFromUri('./models'),
        faceapi.nets.faceExpressionNet.loadFromUri('./models'),
    ]);

    videoFeedEl.srcObject = stream; // Start video after models load

    // Set up canvas
    const canvas = document.getElementById('canvas');
    const rect = videoFeedEl.getBoundingClientRect();
    canvas.style.left = `${rect.left}px`;
    canvas.style.top = `${rect.top}px`;
    canvas.width = rect.width;
    canvas.height = rect.height;

    // Load reference face image
    const refFace = await faceapi.fetchImage('https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Michael_Jordan_in_2014.jpg/220px-Michael_Jordan_in_2014.jpg');
    
    let refFaceAiData = await faceapi
        .detectSingleFace(refFace)
        .withFaceLandmarks()
        .withFaceDescriptor();

    if (!refFaceAiData) {
        console.error("Reference face not detected.");
        return;
    }

    let faceMatcher = new faceapi.FaceMatcher(refFaceAiData);

    // Real-time face detection and matching
    setInterval(async () => {
        let faceAIData = await faceapi
            .detectAllFaces(videoFeedEl)
            .withFaceLandmarks()
            .withFaceDescriptors()
            .withAgeAndGender()
            .withFaceExpressions();

        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
        faceAIData = faceapi.resizeResults(faceAIData, videoFeedEl);
        faceapi.draw.drawDetections(canvas, faceAIData);
        faceapi.draw.drawFaceLandmarks(canvas, faceAIData);
        faceapi.draw.drawFaceExpressions(canvas, faceAIData);

        faceAIData.forEach(face => {
            const { age, gender, genderProbability } = face;
            const genderText = `${gender} - ${Math.round(genderProbability * 100)}%`;
            const ageText = `${Math.round(age)} years`;
            const textField = new faceapi.draw.DrawTextField([genderText, ageText], face.detection.box.topRight);
            textField.draw(canvas);

            // Fix descriptor issue
            let label = faceMatcher.findBestMatch(face.descriptor).toString();
            let options = { label: "Jordan" };
            if (label.includes("unknown")) {
                options = { label: "Unknown subject..." };
            }

            // Fix incorrect detection box reference
            const drawBox = new faceapi.draw.DrawBox(face.detection.box, options);
            drawBox.draw(canvas);
        });
    }, 200);
}

run();
