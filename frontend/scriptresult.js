document.addEventListener('DOMContentLoaded', function(){

const predictionClassLabel = document.getElementById('predictionClass');
const predictionConfidenceLabel = document.getElementById('predictionConfidence');
const predictionJsonLabel = document.getElementById('predictionJSON');
const sourceImageContainer = document.getElementById('sourceImage');
const classifyAnotherButton = document.getElementById('classifyBtn');

let rawPredictionData = null;

const urlParams = new URLSearchParams(window.location.search);
const className = urlParams.get('class');
const confidence = urlParams.get('confidence');
const imageData = sessionStorage.getItem('previewImageData');
const rawJson = JSON.parse(decodeURIComponent(urlParams.get('raw')));

if (className && confidence && predictionClassLabel && predictionConfidenceLabel){
    predictionClassLabel.textContent = `Model predicts: ${className}`;
    const confidencePercent = (parseFloat(confidence) * 100).toFixed(1);
    predictionConfidenceLabel.textContent = `With confidence: ${confidencePercent}`;
} else {
    predictionClassLabel.textContent = 'Model predicts: Error retrieving data.';
    predictionConfidenceLabel.textContent = 'With confidence: Error retrieving data.';
    console.error('Missing class or confidence in the URL parameters.');
}

if(imageData && sourceImageContainer){
    const imgElement = document.createElement('img');
    imgElement.src = imageData;
    imgElement.alt = 'Source image';
    imgElement.style.maxWidth = '224px';
    imgElement.style.maxHeight = '224px';
    sourceImageContainer.appendChild(imgElement);
    sessionStorage.removeItem('previewImageData');
} else {
    sourceImageContainer.textContent = 'Source image preview not available.';
    console.warn('Image data not found in sessionStorage.');
}

if (rawJson && predictionJsonLabel){
    try{
        predictionJsonLabel.style.cursor = 'pointer';
        predictionJsonLabel.style.textDecoration = 'underline';
        predictionJsonLabel.addEventListener('click', () =>{
            alert(JSON.stringify(rawJson, null, 2));
        });
    } catch(e){
        console.error('Error parsing raw JSON data from URL parameters: ', e);
        predictionJsonLabel.textContent = 'Error displaying JSON';
        predictionJsonLabel.style.textDecoration = 'none';
        predictionJsonLabel.style.cursor = 'default';
    }
} else {
    predictionJsonLabel.textContent = '';
}

classifyAnotherButton.addEventListener('click', () =>{
    window.location.href = 'index.html';
});




});