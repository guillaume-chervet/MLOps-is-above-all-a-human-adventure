import React, { useState } from "react";
import axios from "axios";

import "./App.css";


function App() {
  const [file, setFile] = useState(null);
  const [type, setType] = useState(null);
  const [url, setUrl] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [feedback, setFeedback] = useState(null);

  const UPLOAD_ENDPOINT =
    "http://localhost:8064/upload-integration";


  const uploadFile = async file => {
    const formData = new FormData();
    formData.append("file", file);

    const str = JSON.stringify({version:"1.0.0", "type":"opencv", "youhou":"qsdqsdqsdqsd"});
    const bytes = new TextEncoder().encode(str);
    const blob = new Blob([bytes], {
      type: "application/json;charset=utf-8"
    });
    formData.append("settings", blob);


    return await axios.post(UPLOAD_ENDPOINT, formData, {
      headers: {
        "content-type": "multipart/form-data"
      }
    });
  };

  const handleOnChange = async e => {
    const file = e.target.files[0];
    setFile(file);
    const objectURL = URL.createObjectURL(file);
    setUrl(objectURL);
    setPrediction(null);
    setFeedback(null);
    let res = await uploadFile(file);
    setPrediction(res.data)
  };

  const handleFeedback = e => {
    setFeedback(true);
  };

  const handleType = e => {
    setType(e.target.value);
  }

  return (
      <>
    <form>
      <h1>Production environement</h1>
      <input type="file" onChange={handleOnChange} />
      <select  value={type} onChange={handleType} >
        <option value="pillow">Pillow</option>
        <option value="opencv">Opencv</option>
      </select>
    </form>

    { url && <img src={url} alt={"selected image"}  style={{maxWidth: "260px", maxHeight: "260px"}} />}
        {prediction && <>
          <p style={{fontSize:"2em", "padding": "0.4em", "margin": "0em", color:"white", "backgroundColor": "brown", "textAlign": "center"}}>It's a <b>{prediction.prediction}</b></p>


          { feedback === true ? <p>Thank you!</p>  : <> <label>Are you agree with that result ?</label>
        <button onClick={handleFeedback} type="button">Yes</button><button  onClick={handleFeedback} type="button">No</button>
          </>}
        </>
        }


        </>

  );
}

export default App;


