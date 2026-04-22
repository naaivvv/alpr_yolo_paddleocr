document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const videoPreview = document.getElementById('video-preview');
    const btnCapture = document.getElementById('btn-capture');
    const resultImage = document.getElementById('result-image');
    const previewPlaceholder = document.getElementById('preview-placeholder');
    const loader = document.getElementById('loader');
    const resultsBody = document.getElementById('results-body');
    const detectionCount = document.getElementById('detection-count');
    const toast = document.getElementById('toast');
    const toastMessage = document.getElementById('toast-message');

    let stream = null;

    // Tab Switching Logic
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.add('hidden'));
            
            btn.classList.add('active');
            document.getElementById(`tab-${tabId}`).classList.remove('hidden');

            if (tabId === 'camera') {
                initCamera();
            } else {
                stopCamera();
            }
        });
    });

    // File Upload Logic
    dropZone.addEventListener('click', () => fileInput.click());
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            processFile(file);
        }
    });

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) processFile(file);
    });

    // Camera Logic
    async function initCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } } 
            });
            videoPreview.srcObject = stream;
            showToast('Camera initialized');
        } catch (err) {
            console.error('Camera error:', err);
            showToast('Error accessing camera', true);
        }
    }

    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
    }

    btnCapture.addEventListener('click', () => {
        const canvas = document.getElementById('snapshot-canvas');
        canvas.width = videoPreview.videoWidth;
        canvas.height = videoPreview.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoPreview, 0, 0);
        
        canvas.toBlob((blob) => {
            const file = new File([blob], "snapshot.jpg", { type: "image/jpeg" });
            processFile(file);
        }, 'image/jpeg', 0.9);
    });

    // API Interaction
    async function processFile(file) {
        showLoader(true);
        
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/inference', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Inference request failed');

            const data = await response.json();
            updateUI(data);
            showToast('Inference complete');
        } catch (err) {
            console.error('Error:', err);
            showToast('Processing error occurred', true);
        } finally {
            showLoader(false);
        }
    }

    // UI Updates
    function updateUI(data) {
        // Update Image
        resultImage.src = data.image_base64;
        resultImage.classList.remove('hidden');
        previewPlaceholder.classList.add('hidden');

        // Update Counter
        const count = data.results.length;
        detectionCount.textContent = `${count} Detected`;
        detectionCount.className = count > 0 ? 'badge success' : 'badge';

        // Update Table
        resultsBody.innerHTML = '';
        if (data.results.length === 0) {
            resultsBody.innerHTML = '<tr><td colspan="4" class="empty-state">No license plates detected</td></tr>';
            return;
        }

        data.results.forEach(res => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><span class="badge">${res.class_name}</span></td>
                <td style="font-family: monospace; font-weight: bold; letter-spacing: 1px;">${res.text || '---'}</td>
                <td>${(res.conf * 100).toFixed(1)}%</td>
                <td>${res.text_conf > 0 ? (res.text_conf * 100).toFixed(1) + '%' : '---'}</td>
            `;
            resultsBody.appendChild(row);
        });
    }

    function showLoader(show) {
        loader.classList.toggle('hidden', !show);
    }

    function showToast(message, isError = false) {
        toastMessage.textContent = message;
        toast.style.borderColor = isError ? 'var(--danger)' : 'var(--accent)';
        toast.classList.remove('hidden');
        setTimeout(() => toast.classList.add('hidden'), 3000);
    }
});
