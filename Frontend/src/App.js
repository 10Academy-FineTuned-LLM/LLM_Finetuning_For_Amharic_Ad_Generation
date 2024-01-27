import './App.css';
import { useState } from 'react';

function App() {

  const [inputValue, SetInputValue] = useState('');
  const [response, SetResponse] = useState('');

  const handleInputChange = (e) => {
    SetInputValue(e.target.value)
  };
  const handleSubmit = async () => {
    try {
      const response = await fetch('API_END_POINT', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ input: inputValue }),
      });

      const data = await response.json();
      SetResponse(data.response);
    } catch (error) {
      console.error(error);
    }
  };
  
  return (
    <div className="App">
      <header className="App-header">
        <p className='title'>
           AMHARIC AD GENERATOR: 
        </p>
        <div className="input">
          <input value = {inputValue} onChange={handleInputChange}/>
        </div>
        <div className='button'>
          <button onClick={handleSubmit}>Generate AD</button>
        </div>
        {response && (
          <div className='response'>
            <p>{response}</p>
          </div>
        )}
        
      </header>
    </div>
  );
}

export default App;
