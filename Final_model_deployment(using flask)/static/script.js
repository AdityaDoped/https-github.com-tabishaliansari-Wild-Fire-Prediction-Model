function predictWildfire() {
    const year = document.getElementById('year').value;
    const month = document.getElementById('month').value;
    const resultDiv = document.getElementById('result');

    // Simple prediction logic for demonstration
    if (year && month) {
        if ((month >= 6 && month <= 9) || (year % 2 === 0 && month % 2 === 0)) {
            resultDiv.textContent = 'Prediction: Fire is present.';
        } else {
            resultDiv.textContent = 'Prediction: Fire is not present.';
        }
    } else {
        resultDiv.textContent = 'Please enter both year and month.';
    }
}