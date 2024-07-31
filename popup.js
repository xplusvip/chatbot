const serverUrl = 'https://chattbot.onrender.com'; // Replace with your server URL

document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('question-button').addEventListener('click', getAnswer);
});

async function getAnswer() {
    const question = document.getElementById('question').value;

    try {
        const response = await fetch(`${serverUrl}/question`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: question })
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Network response was not ok: ${errorText}`);
        }

        const data = await response.json();
        document.getElementById('answer').innerText = data.answer || data.error;
    } catch (error) {
        console.error('Error:', error);
        alert(`Failed to get answer: ${error.message}`);
    }
}
