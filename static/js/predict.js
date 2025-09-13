$(document).ready(function() {
    // --- Element Variables ---
    const uploadArea = $('#upload-form');
    const fileInput = $('#file-input');
    const browseBtn = $('#browse-btn');
    const fileNameDisplay = $('#file-name-display');
    
    const resultsCard = $('#results-card');
    const placeholder = $('#results-placeholder');
    const loadingSpinner = $('#loading-spinner');
    const predictionOutput = $('#prediction-output');
    
    const resultImage = $('#result-image');
    const breedList = $('#breed-list');
    const resetBtn = $('#reset-btn');

    // --- Core Functions ---

    // Function to handle the file (from either browse or drag-drop)
    function handleFile(file) {
        if (!file) return;

        // Validate file type
        const allowedTypes = ['image/jpeg', 'image/png', 'image/webp'];
        if (!allowedTypes.includes(file.type)) {
            showFlashMessage('Invalid file type. Please upload a JPG, PNG, or WEBP image.', 'error');
            return;
        }

        // Display file name and get ready for prediction
        fileNameDisplay.text(`Selected: ${file.name}`);
        
        // Create FormData and start prediction
        const formData = new FormData();
        formData.append('file', file);
        predict(formData);
    }

    // Function to make AJAX call for prediction
    function predict(formData) {
        // --- UI Changes for Loading State ---
        placeholder.hide();
        predictionOutput.hide();
        loadingSpinner.show();
        uploadArea.addClass('processing');

        $.ajax({
            url: '/api/predict',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                if (response.success) {
                    displayResults(response);
                } else {
                    showFlashMessage(response.error || 'An unknown error occurred.', response.category || 'error');
                    resetToInitialState();
                }
            },
            error: function(jqXHR) {
                const errorResponse = jqXHR.responseJSON || { error: 'A server error occurred. Please try again later.' };
                showFlashMessage(errorResponse.error, 'error');
                resetToInitialState();
            }
        });
    }

    // // Function to display results
    // function displayResults(data) {
    //     // Set the result image
    //     resultImage.attr('src', data.image_url);

    //     // Clear and populate the breed list
    //     breedList.empty();
    //     data.results.forEach((item, index) => {
    //         const isTopResult = index === 0;
    //         const listItem = `
    //             <li class="${isTopResult ? 'top-result' : ''}">
    //                 <div class="breed-name">
    //                     ${isTopResult ? '<i class="fas fa-crown"></i>' : ''}
    //                     ${item.breed}
    //                 </div>
    //                 <div class="breed-score">
    //                     <div class="progress-bar" style="width: ${item.score}%;"></div>
    //                     <span>${item.score}%</span>
    //                 </div>
    //             </li>
    //         `;
    //         breedList.append(listItem);
    //     });

    //     // --- UI Changes for Result State ---
    //     loadingSpinner.hide();
    //     predictionOutput.fadeIn(400);
    //     uploadArea.removeClass('processing');
    // }


    // static/js/predict.js

        function displayResults(data) {
        // Set the result image
        resultImage.attr('src', data.image_url);

        // Clear and populate the breed list
        breedList.empty();
        data.results.forEach((item, index) => {
            const isTopResult = index === 0;
            const listItem = `
                <li class="${isTopResult ? 'top-result' : ''}">
                    <div class="breed-name">
                        ${isTopResult ? '<i class="fas fa-crown"></i>' : ''}
                        ${item.breed}
                    </div>
                    <div class="breed-score">
                        <div class="progress-bar" style="width: ${item.score}%;"></div>
                        <span>${item.score}%</span>
                    </div>
                </li>
            `;
            breedList.append(listItem);
        });

        // --- NEW: Display Milk Yield ---
        const milkYieldContainer = $('#milk-yield-container');
        const milkYieldValue = $('#milk-yield-value');

        if (data.milk_yield) {
            milkYieldValue.text(data.milk_yield);
            milkYieldContainer.show();
        } else {
            milkYieldContainer.hide();
        }

        // --- UI Changes for Result State ---
        loadingSpinner.hide();
        predictionOutput.fadeIn(400);
        uploadArea.removeClass('processing');
        }

    // Function to reset the entire interface
    function resetToInitialState() {
        fileInput.val(''); // Clear the file input
        fileNameDisplay.text('');
        predictionOutput.hide();
        loadingSpinner.hide();
        placeholder.fadeIn(400);
        uploadArea.removeClass('processing');
    }

    // --- Event Listeners ---

    // Browse button triggers file input click
    browseBtn.on('click', () => fileInput.click());

    // Handle file selection from browse
    fileInput.on('change', function() {
        handleFile(this.files[0]);
    });

    // Reset button functionality
    resetBtn.on('click', resetToInitialState);

    // --- Drag and Drop Functionality ---
    uploadArea.on('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        $(this).addClass('drag-over');
    });

    uploadArea.on('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        $(this).removeClass('drag-over');
    });

    uploadArea.on('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        $(this).removeClass('drag-over');
        
        const files = e.originalEvent.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // Prevent browser from opening the file when dropped outside the area
    $(document).on('dragover drop', function(e) {
        e.preventDefault();
    });

    // Helper function to display flash messages (from your script.js)
    function showFlashMessage(message, category) {
        const flashContainer = $('.flash-messages-container');
        if (flashContainer.length === 0) {
            $('main').prepend('<div class="flash-messages-container"></div>');
        }
        const alertHtml = `
            <div class="alert alert-${category} animate-fade-in-down">
                ${message}
                <button class="close-alert">&times;</button>
            </div>`;
        const $alert = $(alertHtml);
        flashContainer.append($alert);
        
        setTimeout(() => {
            $alert.fadeOut(500, function() { $(this).remove(); });
        }, 5000);
    }
});