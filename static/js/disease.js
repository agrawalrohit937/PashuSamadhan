$(document).ready(function() {
    // --- Element Variables ---
    const chatWindow = $('#chat-window');
    const quickRepliesContainer = $('#quick-replies-container');
    const textInputGroup = $('#text-input-group');
    const symptomInput = $('#symptom-input');
    const addSymptomBtn = $('#add-symptom-btn'); // Renamed from symptom-btn to be specific
    const getDiagnosisBtn = $('#get-diagnosis-btn');
    
    // --- State Management ---
    let conversationState = {};
    let allSymptoms = [];

    // --- Autocomplete Setup ---
    $.get('/api/symptoms', function(data) {
        allSymptoms = data;
        symptomInput.autocomplete({
            source: allSymptoms,
        });
    });

    // --- Core Chatbot Functions ---
    function scrollToBottom() {
        chatWindow.scrollTop(chatWindow[0].scrollHeight);
    }

    function addMessage(sender, message, isHtml = false) {
        const messageClass = sender === 'bot' ? 'bot-message' : 'user-message';
        const messageDiv = $(`<div class="chat-message ${messageClass} animate-pop-in"></div>`);
        if (isHtml) {
            messageDiv.html(message);
        } else {
            messageDiv.text(message);
        }
        chatWindow.append(messageDiv);
        scrollToBottom();
    }

    function showQuickReplies(options) {
        quickRepliesContainer.empty().show();
        textInputGroup.hide();
        options.forEach(option => {
            const button = $(`<button class="quick-reply-btn">${option}</button>`);
            button.on('click', () => handleUserInput(option));
            quickRepliesContainer.append(button);
        });
    }

    function setupTextInput(placeholder, inputType = 'text') {
        quickRepliesContainer.hide();
        textInputGroup.css('display', 'flex');
        symptomInput.attr('placeholder', placeholder).attr('type', inputType).val('').focus();
        
        // Temporarily change button functions for age/temp
        addSymptomBtn.text('Next').off('click').on('click', function() {
            handleUserInput(symptomInput.val());
        });
        getDiagnosisBtn.hide(); // Hide diagnose button during age/temp entry
    }
    
    function setupSymptomInput() {
        quickRepliesContainer.hide();
        textInputGroup.css('display', 'flex');
        symptomInput.attr('placeholder', 'Type a symptom...').attr('type', 'text').val('').focus();
        
        addSymptomBtn.html('<i class="fas fa-plus"></i> Add').off('click').on('click', addSymptomHandler);
        getDiagnosisBtn.show();
        updateDiagnosisButton();
    }
    
    function updateDiagnosisButton() {
        getDiagnosisBtn.prop('disabled', conversationState.symptoms.length === 0);
    }
    
    // --- Conversation Flow ---
    function startConversation() {
        conversationState = {
            animal: null,
            age: null,
            temperature: null,
            symptoms: [],
            currentStep: 'animal'
        };
        chatWindow.empty();
        addMessage('bot', 'Namaste! Main Pashu Vaidya AI hu. Main aapke pashu ki beemari ka pata lagane mein madad karunga.');
        setTimeout(() => {
            addMessage('bot', 'Aap kis jaanwar ke liye jaankari chahte hain?');
            showQuickReplies(['गाय (Cow)', 'भैंस (Buffalo)']);
        }, 1000);
    }

    function handleUserInput(input) {
        addMessage('user', input);
        
        switch (conversationState.currentStep) {
            case 'animal':
                conversationState.animal = input.includes('Cow') ? 'cow' : 'buffalo';
                conversationState.currentStep = 'age';
                setTimeout(() => {
                    addMessage('bot', 'Theek hai. Ab kripya jaanwar ki umra (saal mein) batayein.');
                    setupTextInput('Enter age in years...', 'number');
                }, 500);
                break;
                
            case 'age':
                const age = parseFloat(input);
                if (isNaN(age) || age <= 0) {
                    addMessage('bot', 'Kripya ek valid umra (0 se zyada) darj karein.');
                    setupTextInput('Enter age in years...', 'number');
                    return;
                }
                conversationState.age = age;
                conversationState.currentStep = 'temperature';
                setTimeout(() => {
                    addMessage('bot', 'Dhanyavaad. Ab jaanwar ka sharirik taapmaan (Fahrenheit mein) batayein.');
                    setupTextInput('Enter temperature in °F...', 'number');
                }, 500);
                break;
                
            case 'temperature':
                const temp = parseFloat(input);
                if (isNaN(temp) || temp <= 0) {
                    addMessage('bot', 'Kripya ek valid taapmaan darj karein.');
                    setupTextInput('Enter temperature in °F...', 'number');
                    return;
                }
                conversationState.temperature = temp;
                conversationState.currentStep = 'symptoms';
                setTimeout(() => {
                    addMessage('bot', 'Ab kripya mukhya lakshan (symptoms) batayein. Kam se kam 1-3 lakshan add karein.');
                    setupSymptomInput();
                }, 500);
                break;
        }
    }
    
    function addSymptomHandler() {
        const symptom = symptomInput.val().trim();
        if (symptom && allSymptoms.includes(symptom) && !conversationState.symptoms.includes(symptom)) {
            conversationState.symptoms.push(symptom);
            addMessage('user', `Added Symptom: ${symptom}`);
            symptomInput.val('').focus();
            if (conversationState.symptoms.length >= 3) {
                 addMessage('bot', 'Aapne paryaapt lakshan add kar diye hain. Ab aap "Diagnose" par click kar sakte hain.');
            }
        } else if (!symptom) {
             addMessage('bot', 'Please type a symptom.');
        } else if (conversationState.symptoms.includes(symptom)) {
            addMessage('bot', 'Yeh lakshan pehle se hi add kiya ja chuka hai.');
        } else {
            addMessage('bot', 'Kripya list mein se ek valid lakshan chunein.');
        }
        updateDiagnosisButton();
    }
    
    getDiagnosisBtn.on('click', function() {
        addMessage('bot', 'Dhanyavaad. Main sabhi jaankari ke aadhar par vishleshan (analysis) kar raha hu...');
        quickRepliesContainer.hide();
        textInputGroup.hide();
        
        $.ajax({
            url: '/api/diagnose',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                animal: conversationState.animal,
                age: conversationState.age,
                temperature: conversationState.temperature,
                symptoms: conversationState.symptoms
            }),
            success: function(response) {
                if(response.success) {
                    const resultHtml = `
                        <p><strong>Nishkarsh (Diagnosis):</strong> ${response.disease}</p>
                        <p><strong>Confidence Score:</strong> ${response.confidence}%</p>
                        <hr>
                        <p><strong>Mukhya Kaaran (Key Factors):</strong> Yeh nishkarsh aapke diye gaye in lakshano par aadharit hai: <em>${response.key_symptoms.join(', ')}</em>.</p>
                        <p class="disclaimer"><strong>Salah:</strong> Yeh ek preliminary jaanch hai. Sahi ilaaj ke liye turant pashu chikitsak se sampark karein.</p>
                    `;
                    addMessage('bot', resultHtml, true);
                } else {
                    addMessage('bot', `Ek error aayi: ${response.error}`);
                }
                setTimeout(() => showQuickReplies(['Nayi Jaanch Shuru Karein']), 2000);
                conversationState.currentStep = 'end';
            },
            error: function() {
                addMessage('bot', 'Server se sampark nahi ho pa raha hai. Kripya baad mein prayas karein.');
                setTimeout(() => showQuickReplies(['Nayi Jaanch Shuru Karein']), 2000);
                conversationState.currentStep = 'end';
            }
        });
    });
    
    // To handle "Start New Diagnosis" and initial animal selection
    quickRepliesContainer.on('click', '.quick-reply-btn', function() {
        const text = $(this).text();
        if (text === 'Nayi Jaanch Shuru Karein') {
            startConversation();
        } else if (conversationState.currentStep === 'animal') {
            handleUserInput(text);
        }
    });

    // --- Start the chatbot on page load ---
    startConversation();
});