$(document).ready(function() {
    // Smooth Scrolling for anchor links (if any)
    $('a[href^="#"]').on('click', function(event) {
        var target = $(this.attr('href'));
        if (target.length) {
            event.preventDefault();
            $('html, body').animate({
                scrollTop: target.offset().top - $('.main-header').outerHeight() // Adjust for sticky header
            }, 800);
        }
    });

    // Mobile Navigation Toggle
    $('.nav-toggle').on('click', function() {
        $('.nav-links').toggleClass('active');
    });

    // Close Flash Messages
    $(document).on('click', '.close-alert', function() {
        $(this).closest('.alert').fadeOut(300, function() {
            $(this).remove();
        });
    });

    // --- Image Upload Preview & Clear ---
    const fileUpload = $('#file-upload');
    const fileNameSpan = $('#file-name');
    const previewImage = $('#preview-image');
    const imagePreviewContainer = $('#image-preview-container');
    const predictButton = $('#predict-button');
    const clearImageButton = $('#clear-image');

    fileUpload.on('change', function() {
        const file = this.files[0];
        if (file) {
            fileNameSpan.text(file.name);
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.attr('src', e.target.result);
                imagePreviewContainer.fadeIn();
                predictButton.prop('disabled', false); // Enable button
            };
            reader.readAsDataURL(file);
        } else {
            fileNameSpan.text('No file chosen');
            previewImage.attr('src', '#');
            imagePreviewContainer.fadeOut();
            predictButton.prop('disabled', true); // Disable button
        }
    });

    clearImageButton.on('click', function() {
        fileUpload.val(''); // Clear file input
        fileUpload.trigger('change'); // Trigger change to update preview and button
        $('#prediction-results-container').fadeOut(); // Also hide results if visible
    });

    // --- AJAX Prediction ---
    $('#upload-form').on('submit', function(e) {
        e.preventDefault(); // Prevent default form submission

        const formData = new FormData(this);
        const loadingSpinner = $('#loading-spinner');
        const predictionResultsContainer = $('#prediction-results-container');
        const breedList = $('#breed-list');
        const resultCattleImage = $('#result-cattle-image');
        
        // Hide previous results, show spinner
        predictionResultsContainer.hide();
        loadingSpinner.fadeIn();
        predictButton.prop('disabled', true).html('<i class="fas fa-spinner fa-spin"></i> Predicting...');

        $.ajax({
            url: '/api/predict', // Your Flask API endpoint
            type: 'POST',
            data: formData,
            processData: false, // Don't process the data, formData is already correctly formatted
            contentType: false, // Don't set content type, let jQuery do it for FormData
            success: function(response) {
                loadingSpinner.fadeOut();
                predictButton.prop('disabled', false).html('<i class="fas fa-cogs"></i> Predict Breed');

                if (response.success) {
                    resultCattleImage.attr('src', response.image_url);
                    breedList.empty(); // Clear previous predictions
                    response.results.forEach(function(item) {
                        breedList.append(`
                            <li>
                                <strong>${item.breed}:</strong> <span class="score">${item.score}%</span>
                            </li>
                        `);
                    });
                    predictionResultsContainer.fadeIn();
                    // Scroll to results
                    $('html, body').animate({
                        scrollTop: predictionResultsContainer.offset().top - $('.main-header').outerHeight() - 20
                    }, 800);

                } else {
                    // Display error using flash message like style
                    showFlashMessage(response.error, response.category || 'error');
                }
            },
            error: function(jqXHR, textStatus, errorThrown) {
                loadingSpinner.fadeOut();
                predictButton.prop('disabled', false).html('<i class="fas fa-cogs"></i> Predict Breed');
                const errorResponse = jqXHR.responseJSON || { error: 'An unexpected error occurred. Please try again.', category: 'error' };
                showFlashMessage(errorResponse.error, errorResponse.category);
            }
        });
    });

    // Helper function to display custom flash messages
    function showFlashMessage(message, category) {
        const flashContainer = $('.flash-messages-container');
        if (flashContainer.length === 0) {
            $('main').prepend('<div class="flash-messages-container"></div>');
        }
        flashContainer.append(`
            <div class="alert alert-${category} animate-fade-in-down">
                ${message}
                <button class="close-alert">&times;</button>
            </div>
        `);
        // Auto-hide after some time
        $('.alert:last').delay(5000).fadeOut(500, function() {
            $(this).remove();
        });
    }


    // --- Animate elements on scroll ---
    const animateOnScrollElements = $('.animate-on-scroll, .animate-slide-up, .animate-slide-left, .animate-slide-right');

    function checkAnimation() {
        const windowHeight = $(window).height();
        animateOnScrollElements.each(function() {
            const elementOffset = $(this).offset().top;
            const scrollPos = $(window).scrollTop();
            const elementHeight = $(this).outerHeight();

            // When element is in viewport
            if (scrollPos + windowHeight > elementOffset + elementHeight / 4 && scrollPos < elementOffset + elementHeight) {
                $(this).addClass('is-visible');
            } else {
                // Optionally remove class if you want elements to re-animate on scroll back up
                // $(this).removeClass('is-visible');
            }
        });
    }

    // Run on scroll and on load
    $(window).on('scroll resize', checkAnimation);
    checkAnimation(); // Initial check on load
});