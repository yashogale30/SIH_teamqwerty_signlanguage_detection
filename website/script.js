const video = document.getElementById('video');
const recognizedText = document.getElementById('recognized-text');

// Access the webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => {
        console.error("Error accessing the camera: ", err);
    });

// Load TensorFlow model (if using TensorFlow.js)
// Make sure to include the TensorFlow.js script in the HTML or load it here dynamically
async function loadModel() {
    const model = await tf.loadLayersModel('path_to_model/model.json');

    // Process video frames in real-time
    video.addEventListener('loadeddata', () => {
        setInterval(() => {
            const frame = tf.browser.fromPixels(video);
            const prediction = model.predict(frame.expandDims(0));

            // Display the recognized text (use the model’s prediction)
            recognizedText.innerText = prediction; // Adjust based on the model’s output
        }, 100);
    });
}

// Call the loadModel function
loadModel();

const cameraOnBtn = document.getElementById('camera-on');
const cameraOffBtn = document.getElementById('camera-off');
let stream;

// Function to start the camera
function startCamera() {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(mediaStream => {
            stream = mediaStream;
            video.srcObject = stream;
        })
        .catch(err => {
            console.error("Error accessing the camera: ", err);
        });
}

// Function to stop the camera
function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
    }
}

// Event listeners for buttons
cameraOnBtn.addEventListener('click', startCamera);
cameraOffBtn.addEventListener('click', stopCamera);

// Start the camera automatically on page load
startCamera();
