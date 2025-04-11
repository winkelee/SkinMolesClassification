document.addEventListener("DOMContentLoaded", function(){
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const classifyButton = document.getElementById("classifyBtn");

const API_PREDICT_ENDPOINT = "http://127.0.0.1:8000/predict/"

let selectedFile = null;
let selectedFileBase64 = null;

dropzone.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    const files = e.target.files;
    if (files.length > 0) {
        handleFileSelection(files[0]); 
    }
});

dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.textContent = 'Dragging a file detected!';
    dropzone.style.backgroundColor = '#c8d2cb';
});

dropzone.addEventListener('dragleave', () => {
    dropzone.textContent = 'Drag or choose file here...';
    dropzone.style.backgroundColor = '#CDD6D0'
})

dropzone.addEventListener('drop', (e) => {
    console.log('Drop event executed.');
    e.preventDefault();
    const files = e.dataTransfer.files;
    if(files.length > 0){
        if (files[0].type.startsWith('image/')){
            handleFileSelection(files[0]);
        }

        else{
            alert('Please select an image file.')
            clearSelection();
        }
    }
});

classifyButton.addEventListener('click', () => {
    if (selectedFile !== null){
        uploadAndPredict(selectedFile);
    } else{
        alert('Please select or drop an image before classifying it.')
    }
});


function handleFileSelection(file){
    selectedFile = file;
    dropzone.textContent = `Selected: ${file.name}`;
    
    const reader = new FileReader();
    reader.onload = function(e){
        selectedFileBase64 = e.target.result;
    }
    reader.readAsDataURL(file);
    reader.onerror = function(){
        alert('Something wnt wrong while reading file for preview');
        clearSelection();
    }
}

function clearSelection() {
    selectedFile = null;
    selectedFileBase64 = null;
    fileInput.value = null;
    dropzone.textContent = 'Drag or choose file here...';
}

function uploadAndPredict(file){
    if(!selectedFileBase64){
        alert('Still processing file for preview. Please, wait');
        return;
    }
    dropzone.textContent = 'Uploading and predicting...';
    classifyButton.disabled = true;

    const formData = new FormData();
    formData.append('file', file, file.name);
    fetch(API_PREDICT_ENDPOINT, {
        method: 'POST',
        body: formData
    })
    .then(response =>{
        if(!response.ok){
            response.json().then(errData => {
                throw new Error(errData.detail || `API ERROR ${response.statusText} (STATUS: ${response.status})`);
            }).catch(() => {
                throw new Error(`API ERROR: ${response.statusText} STATUS: ${response.status}`);
            })
        }
        return response.json();
    }).then(data =>{
        console.log('API success: ', data);
        try{
            sessionStorage.removeItem('previewImageData');
            sessionStorage.setItem('previewImageData', selectedFileBase64);
        } catch(e){
            console.error("Error saving image for preview", e);
        }
        const predictionData = data.prediction;
        const className = encodeURIComponent(predictionData.predicted_class);
        const confidence = encodeURIComponent(predictionData.confidence.toFixed(4));
        const rawJson = encodeURIComponent(JSON.stringify(data)); // Stringifying DATA, the whole response with useful info.
        window.location.href = `result.html?class=${className}&confidence=${confidence}&raw=${rawJson}`;
    }).catch(error => {
        console.error('Prediction failed', error);
        alert(`Prediction failed: ${error.message}`);
        clearSelection();
        classifyButton.disabled = false;
    });
}



});