document.addEventListener('DOMContentLoaded', () => {
    const promptInput = document.getElementById('promptInput');
    const maxLength = document.getElementById('maxLength');
    const generateBtn = document.getElementById('generateBtn');
    const btnText = generateBtn.querySelector('.btn-text');
    const loader = generateBtn.querySelector('.loader');
    const resultsContainer = document.getElementById('resultsContainer');
    const resultsList = document.getElementById('resultsList');

    const API_URL = '/generate';

    const setLoader = (loading) => {
        generateBtn.disabled = loading;
        if (loading) {
            btnText.classList.add('hidden');
            loader.classList.remove('hidden');
        } else {
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
        }
    };

    const generateText = async () => {
        const prompt = promptInput.value.trim();
        if (!prompt) {
            alert('الرجاء كتابة نص للبدء');
            return;
        }

        setLoader(true);

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: prompt,
                    max_length: parseInt(maxLength.value) || 100,
                    num_sequences: 1
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'فشل في توليد النص بالدارجة');
            }

            const data = await response.json();

            // Show results
            resultsContainer.classList.remove('hidden');

            // Clear previous results or prepend for chat-like feel? 
            // Here we clear to show the best output clearly
            resultsList.innerHTML = '';

            data.generated_texts.forEach(text => {
                const div = document.createElement('div');
                div.className = 'result-item';
                div.textContent = text;
                resultsList.appendChild(div);
            });

            // Smooth scroll to results
            resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });

        } catch (error) {
            console.error('Error:', error);
            alert(`خطأ: ${error.message}`);
        } finally {
            setLoader(false);
        }
    };

    generateBtn.addEventListener('click', generateText);

    // Allow Ctrl+Enter to submit
    promptInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            generateText();
        }
    });
});
