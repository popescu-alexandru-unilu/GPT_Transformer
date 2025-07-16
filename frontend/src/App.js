import React, { useState } from 'react';
import './App.css';

function App() {
  const [inputText, setInputText] = useState('');
  const [resultText, setResultText] = useState('');

  const handleGenerateClick = async () => {
    const response = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: inputText })
    });
    const data = await response.json();
    setResultText(data.generated_text);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>GPT-3 Interface</h1>
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          rows="10"
          cols="80"
        />
        <br />
        <button onClick={handleGenerateClick}>Generate</button>
        <br />
        <h2>Generated Text:</h2>
        <p>{resultText}</p>
      </header>
    </div>
  );
}

export default App;
