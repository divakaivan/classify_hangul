var labels = ['a', 'ae', 'b', 'bb', 'ch', 'd', 
                'e', 'eo', 'eu', 'g', 'gg', 'h', 
                'i', 'j', 'k', 'm', 'n', 'ng', 'o', 
                'p', 'r', 's', 'ss', 't', 'u', 'ya', 
                'yae', 'ye', 'yo', 'yu'];

var canvas, ctx, clearButton, saveImageButton, classifyButton;
var pos = {x:0, y:0};
var model;

function setPosition(e){
    pos.x = e.clientX - canvas.offsetLeft;
    pos.y = e.clientY - canvas.offsetTop;
}
    
function draw(e) {
    if(e.buttons !== 1) return;
    ctx.beginPath();
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';
    ctx.moveTo(pos.x, pos.y);
    setPosition(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
}
    
function erase() {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

async function classifyImage() {
    if (!model) {
        console.error('Model is not loaded.');
        return;
    }

    try {
        // Preprocess the drawn image
        var inputTensor = await preprocessImage();
        // Make predictions using the loaded model
        var prediction = await model.predict(inputTensor).data();
        console.log('Prediction:', prediction);

        let maxIndex = prediction.indexOf(Math.max(...prediction));

        // Get the corresponding label
        let predictedLabel = labels[maxIndex];
        document.getElementById('prediction').textContent = 'Predicted label: ' + predictedLabel;
        console.log(predictedLabel);
    } catch (error) {
        console.error('Error during classification:', error);
    }
}

async function preprocessImage() {
    var dataURL = canvas.toDataURL('image/png');
    var image = new Image();
    image.src = dataURL;

    return new Promise((resolve, reject) => {   
        image.onload = function() {
            var tensor = tf.browser.fromPixels(image)
                .resizeNearestNeighbor([64, 64]) // Resize to match model's input shape
                .mean(2)
                .toFloat();
            
            // Ensure that the image has 3 color channels (RGB)
            if (tensor.shape[2] === 4) { // Check if image has alpha channel
                tensor = tensor.slice([0, 0, 0], [64, 64, 1]); // Remove alpha channel
            }
            console.log(tensor.shape)
            tensor = tensor.expandDims(0); // Add batch dimension
            
            console.log('Preprocessed image shape:', tensor.shape); // Log the shape
            
            resolve(tensor.div(255.0)); // Normalize pixel values
        };

        image.onerror = function() {
            reject(new Error('Failed to load image'));
        };
    });
}


async function loadModel() {
    try {
        model = await tf.loadLayersModel('hangul_model1.json');
        console.log('Model loaded successfully');
        console.log(model.summary());
    } catch (error) {
        console.error('Error loading model:', error);
    }
}

    
async function init() {
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext("2d");
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mousedown", setPosition);
    canvas.addEventListener("mouseenter", setPosition);
    
    clearButton = document.getElementById('cb');
    clearButton.addEventListener("click", erase);

    classifyButton = document.getElementById('classifyBtn');
    classifyButton.addEventListener("click", classifyImage);

    try {
        await loadModel(); // Wait for the model to load before proceeding
    } catch (error) {
        console.error('Error loading model:', error);
    }
}
