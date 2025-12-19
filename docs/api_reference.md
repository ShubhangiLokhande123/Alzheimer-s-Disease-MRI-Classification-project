\# API Reference



Complete reference for the Alzheimer's Disease Classification REST API.



\## Table of Contents

\- \[Overview](#overview)

\- \[Authentication](#authentication)

\- \[Base URL](#base-url)

\- \[Endpoints](#endpoints)

\- \[Request/Response Formats](#requestresponse-formats)

\- \[Error Handling](#error-handling)

\- \[Rate Limiting](#rate-limiting)

\- \[Code Examples](#code-examples)



---



\## Overview



The SETNN API provides RESTful endpoints for Alzheimer's disease classification from MRI scans. The API is built with FastAPI and supports:



\- Single image prediction

\- Batch processing

\- Model information retrieval

\- Detailed prediction explanations



\### API Features



✅ \*\*High Performance:\*\* Sub-second inference time  

✅ \*\*Scalable:\*\* Handles concurrent requests  

✅ \*\*Secure:\*\* Input validation and error handling  

✅ \*\*Well-Documented:\*\* Interactive Swagger UI  

✅ \*\*Easy Integration:\*\* Standard REST endpoints  



---



\## Authentication



\### Current Version (Development)



No authentication required for development environment.



\### Production Version (Future)



Production API will use JWT Bearer tokens:



```bash

curl -X POST "https://api.example.com/predict" \\

&nbsp; -H "Authorization: Bearer YOUR\_API\_TOKEN" \\

&nbsp; -F "file=@scan.nii.gz"

```



To obtain an API token:

```bash

POST /auth/token

Content-Type: application/json



{

&nbsp; "username": "your\_username",

&nbsp; "password": "your\_password"

}

```



---



\## Base URL



\### Local Development

```

http://localhost:8000

```



\### Production (Example)

```

https://api.alzheimers-classification.com

```



---



\## Endpoints



\### 1. Root Endpoint



\*\*GET /\*\* - API Information



Returns basic API information.



\*\*Request:\*\*

```bash

GET /

```



\*\*Response:\*\*

```json

{

&nbsp; "message": "Alzheimer's Disease Classification API",

&nbsp; "version": "1.0.0",

&nbsp; "docs": "/docs"

}

```



\*\*Status Codes:\*\*

\- `200 OK` - Success



---



\### 2. Health Check



\*\*GET /health\*\* - API Health Status



Check if the API and model are loaded and ready.



\*\*Request:\*\*

```bash

GET /health

```



\*\*Response:\*\*

```json

{

&nbsp; "status": "healthy",

&nbsp; "model\_loaded": true,

&nbsp; "version": "1.0.0"

}

```



\*\*Status Codes:\*\*

\- `200 OK` - API is healthy

\- `503 Service Unavailable` - Model not loaded



\*\*Use Case:\*\*

\- Kubernetes health checks

\- Monitoring systems

\- Load balancer health probes



---



\### 3. Model Information



\*\*GET /model/info\*\* - Get Model Details



Retrieve information about the loaded SETNN model.



\*\*Request:\*\*

```bash

GET /model/info

```



\*\*Response:\*\*

```json

{

&nbsp; "model\_type": "SETNN (Stacked Ensemble Transfer Neural Network)",

&nbsp; "base\_models": \["VGG16", "InceptionV3", "MobileNetV2"],

&nbsp; "num\_classes": 3,

&nbsp; "class\_names": \[

&nbsp;   "Non\_Demented",

&nbsp;   "Mild\_Cognitive\_Impairment",

&nbsp;   "Alzheimers\_Disease"

&nbsp; ],

&nbsp; "input\_shape": \[224, 224, 3]

}

```



\*\*Status Codes:\*\*

\- `200 OK` - Success

\- `503 Service Unavailable` - Model not loaded



---



\### 4. Get Classes



\*\*GET /classes\*\* - List Supported Classes



Retrieve the list of classification classes.



\*\*Request:\*\*

```bash

GET /classes

```



\*\*Response:\*\*

```json

{

&nbsp; "classes": \[

&nbsp;   "Non\_Demented",

&nbsp;   "Mild\_Cognitive\_Impairment",

&nbsp;   "Alzheimers\_Disease"

&nbsp; ],

&nbsp; "num\_classes": 3

}

```



\*\*Status Codes:\*\*

\- `200 OK` - Success

\- `503 Service Unavailable` - Model not loaded



---



\### 5. Single Prediction



\*\*POST /predict\*\* - Predict Single MRI Scan



Classify a single MRI scan and return the predicted diagnosis.



\*\*Request:\*\*

```bash

POST /predict

Content-Type: multipart/form-data



file: <binary\_nifti\_file>

```



\*\*cURL Example:\*\*

```bash

curl -X POST "http://localhost:8000/predict" \\

&nbsp; -H "accept: application/json" \\

&nbsp; -H "Content-Type: multipart/form-data" \\

&nbsp; -F "file=@/path/to/mri\_scan.nii.gz"

```



\*\*Python Example:\*\*

```python

import requests



url = "http://localhost:8000/predict"

files = {"file": open("mri\_scan.nii.gz", "rb")}



response = requests.post(url, files=files)

print(response.json())

```



\*\*Response:\*\*

```json

{

&nbsp; "predicted\_class": "Non\_Demented",

&nbsp; "predicted\_label": 0,

&nbsp; "confidence": 0.9847,

&nbsp; "probabilities": {

&nbsp;   "Non\_Demented": 0.9847,

&nbsp;   "Mild\_Cognitive\_Impairment": 0.0123,

&nbsp;   "Alzheimers\_Disease": 0.0030

&nbsp; },

&nbsp; "above\_threshold": true,

&nbsp; "inference\_time\_seconds": 0.234

}

```



\*\*Response Fields:\*\*



| Field | Type | Description |

|-------|------|-------------|

| `predicted\_class` | string | Predicted class name |

| `predicted\_label` | integer | Class index (0, 1, or 2) |

| `confidence` | float | Confidence score (0-1) |

| `probabilities` | object | Probability for each class |

| `above\_threshold` | boolean | Whether confidence exceeds threshold |

| `inference\_time\_seconds` | float | Time taken for prediction |



\*\*Status Codes:\*\*

\- `200 OK` - Successful prediction

\- `400 Bad Request` - Invalid file format

\- `500 Internal Server Error` - Prediction failed

\- `503 Service Unavailable` - Model not loaded



\*\*Validation:\*\*

\- File must be in NIfTI format (.nii or .nii.gz)

\- File size must be reasonable (<100MB)



---



\### 6. Batch Prediction



\*\*POST /predict/batch\*\* - Predict Multiple MRI Scans



Process multiple MRI scans in a single request.



\*\*Request:\*\*

```bash

POST /predict/batch

Content-Type: multipart/form-data



files: <binary\_nifti\_file\_1>

files: <binary\_nifti\_file\_2>

...

```



\*\*cURL Example:\*\*

```bash

curl -X POST "http://localhost:8000/predict/batch" \\

&nbsp; -F "files=@scan1.nii.gz" \\

&nbsp; -F "files=@scan2.nii.gz" \\

&nbsp; -F "files=@scan3.nii.gz"

```



\*\*Python Example:\*\*

```python

import requests



url = "http://localhost:8000/predict/batch"

files = \[

&nbsp;   ("files", open("scan1.nii.gz", "rb")),

&nbsp;   ("files", open("scan2.nii.gz", "rb")),

&nbsp;   ("files", open("scan3.nii.gz", "rb"))

]



response = requests.post(url, files=files)

print(response.json())

```



\*\*Response:\*\*

```json

{

&nbsp; "total": 3,

&nbsp; "successful": 3,

&nbsp; "failed": 0,

&nbsp; "results": \[

&nbsp;   {

&nbsp;     "filename": "scan1.nii.gz",

&nbsp;     "status": "success",

&nbsp;     "predicted\_class": "Non\_Demented",

&nbsp;     "confidence": 0.9847,

&nbsp;     "probabilities": {

&nbsp;       "Non\_Demented": 0.9847,

&nbsp;       "Mild\_Cognitive\_Impairment": 0.0123,

&nbsp;       "Alzheimers\_Disease": 0.0030

&nbsp;     }

&nbsp;   },

&nbsp;   {

&nbsp;     "filename": "scan2.nii.gz",

&nbsp;     "status": "success",

&nbsp;     "predicted\_class": "Mild\_Cognitive\_Impairment",

&nbsp;     "confidence": 0.8932,

&nbsp;     "probabilities": {

&nbsp;       "Non\_Demented": 0.0542,

&nbsp;       "Mild\_Cognitive\_Impairment": 0.8932,

&nbsp;       "Alzheimers\_Disease": 0.0526

&nbsp;     }

&nbsp;   },

&nbsp;   {

&nbsp;     "filename": "scan3.nii.gz",

&nbsp;     "status": "success",

&nbsp;     "predicted\_class": "Alzheimers\_Disease",

&nbsp;     "confidence": 0.9654,

&nbsp;     "probabilities": {

&nbsp;       "Non\_Demented": 0.0123,

&nbsp;       "Mild\_Cognitive\_Impairment": 0.0223,

&nbsp;       "Alzheimers\_Disease": 0.9654

&nbsp;     }

&nbsp;   }

&nbsp; ]

}

```



\*\*Limitations:\*\*

\- Maximum 50 files per batch request

\- Each file must be <100MB



\*\*Status Codes:\*\*

\- `200 OK` - Batch processed (check individual results)

\- `400 Bad Request` - Invalid request (no files, too many files)

\- `500 Internal Server Error` - Batch processing failed



---



\### 7. Explain Prediction



\*\*POST /predict/explain\*\* - Detailed Prediction Explanation



Get detailed explanation including predictions from individual base models.



\*\*Request:\*\*

```bash

POST /predict/explain

Content-Type: multipart/form-data



file: <binary\_nifti\_file>

```



\*\*cURL Example:\*\*

```bash

curl -X POST "http://localhost:8000/predict/explain" \\

&nbsp; -F "file=@scan.nii.gz"

```



\*\*Response:\*\*

```json

{

&nbsp; "predicted\_class": "Non\_Demented",

&nbsp; "predicted\_label": 0,

&nbsp; "confidence": 0.9847,

&nbsp; "probabilities": {

&nbsp;   "Non\_Demented": 0.9847,

&nbsp;   "Mild\_Cognitive\_Impairment": 0.0123,

&nbsp;   "Alzheimers\_Disease": 0.0030

&nbsp; },

&nbsp; "above\_threshold": true,

&nbsp; "inference\_time\_seconds": 0.345,

&nbsp; "base\_model\_predictions": {

&nbsp;   "vgg16": {

&nbsp;     "predicted\_class": "Non\_Demented",

&nbsp;     "confidence": 0.9723,

&nbsp;     "probabilities": {

&nbsp;       "Non\_Demented": 0.9723,

&nbsp;       "Mild\_Cognitive\_Impairment": 0.0189,

&nbsp;       "Alzheimers\_Disease": 0.0088

&nbsp;     }

&nbsp;   },

&nbsp;   "inceptionv3": {

&nbsp;     "predicted\_class": "Non\_Demented",

&nbsp;     "confidence": 0.9856,

&nbsp;     "probabilities": {

&nbsp;       "Non\_Demented": 0.9856,

&nbsp;       "Mild\_Cognitive\_Impairment": 0.0098,

&nbsp;       "Alzheimers\_Disease": 0.0046

&nbsp;     }

&nbsp;   },

&nbsp;   "mobilenetv2": {

&nbsp;     "predicted\_class": "Non\_Demented",

&nbsp;     "confidence": 0.9612,

&nbsp;     "probabilities": {

&nbsp;       "Non\_Demented": 0.9612,

&nbsp;       "Mild\_Cognitive\_Impairment": 0.0234,

&nbsp;       "Alzheimers\_Disease": 0.0154

&nbsp;     }

&nbsp;   }

&nbsp; }

}

```



\*\*Use Case:\*\*

\- Model debugging

\- Understanding predictions

\- Research and analysis

\- Confidence assessment



\*\*Status Codes:\*\*

\- `200 OK` - Success

\- `400 Bad Request` - Invalid file

\- `500 Internal Server Error` - Explanation failed



---



\## Request/Response Formats



\### Supported File Formats



\*\*Input:\*\*

\- `.nii` - Uncompressed NIfTI

\- `.nii.gz` - Compressed NIfTI (recommended)



\*\*Output:\*\*

\- `application/json` - All responses in JSON format



\### Content Types



\*\*Request:\*\*

```

Content-Type: multipart/form-data

```



\*\*Response:\*\*

```

Content-Type: application/json

```



---



\## Error Handling



\### Error Response Format



```json

{

&nbsp; "detail": "Error message describing what went wrong"

}

```



\### Common Errors



\#### 400 Bad Request

```json

{

&nbsp; "detail": "Invalid file format. Please upload NIfTI file (.nii or .nii.gz)"

}

```



\*\*Causes:\*\*

\- Wrong file format

\- Missing required fields

\- Invalid parameters



\#### 404 Not Found

```json

{

&nbsp; "detail": "Endpoint not found"

}

```



\*\*Causes:\*\*

\- Incorrect URL

\- Typo in endpoint



\#### 500 Internal Server Error

```json

{

&nbsp; "detail": "Prediction failed: <error\_details>"

}

```



\*\*Causes:\*\*

\- Corrupted file

\- Processing error

\- Model inference failure



\#### 503 Service Unavailable

```json

{

&nbsp; "detail": "Model not loaded"

}

```



\*\*Causes:\*\*

\- API starting up

\- Model loading failed

\- Service maintenance



---



\## Rate Limiting



\### Current Limits (Development)



No rate limits in development mode.



\### Production Limits (Future)



| Tier | Requests/Minute | Requests/Day |

|------|----------------|--------------|

| Free | 10 | 100 |

| Basic | 60 | 1000 |

| Pro | 300 | 10000 |



\### Rate Limit Headers



```

X-RateLimit-Limit: 60

X-RateLimit-Remaining: 45

X-RateLimit-Reset: 1640000000

```



\### Rate Limit Exceeded Response



```json

{

&nbsp; "detail": "Rate limit exceeded. Try again in 60 seconds."

}

```



\*\*Status Code:\*\* `429 Too Many Requests`



---



\## Code Examples



\### Python



\#### Simple Prediction



```python

import requests



def predict\_alzheimers(file\_path):

&nbsp;   url = "http://localhost:8000/predict"

&nbsp;   

&nbsp;   with open(file\_path, 'rb') as f:

&nbsp;       files = {'file': f}

&nbsp;       response = requests.post(url, files=files)

&nbsp;   

&nbsp;   if response.status\_code == 200:

&nbsp;       result = response.json()

&nbsp;       print(f"Prediction: {result\['predicted\_class']}")

&nbsp;       print(f"Confidence: {result\['confidence']:.2%}")

&nbsp;       return result

&nbsp;   else:

&nbsp;       print(f"Error: {response.json()\['detail']}")

&nbsp;       return None



\# Usage

result = predict\_alzheimers("path/to/mri\_scan.nii.gz")

```



\#### Batch Prediction



```python

import requests

from pathlib import Path



def predict\_batch(directory):

&nbsp;   url = "http://localhost:8000/predict/batch"

&nbsp;   

&nbsp;   # Get all NIfTI files

&nbsp;   files = \[]

&nbsp;   for file\_path in Path(directory).glob("\*.nii\*"):

&nbsp;       files.append(('files', open(file\_path, 'rb')))

&nbsp;   

&nbsp;   response = requests.post(url, files=files)

&nbsp;   

&nbsp;   # Close files

&nbsp;   for \_, f in files:

&nbsp;       f.close()

&nbsp;   

&nbsp;   if response.status\_code == 200:

&nbsp;       results = response.json()

&nbsp;       print(f"Processed: {results\['successful']}/{results\['total']}")

&nbsp;       return results

&nbsp;   else:

&nbsp;       print(f"Error: {response.json()\['detail']}")

&nbsp;       return None



\# Usage

results = predict\_batch("path/to/mri\_scans/")

```



\#### Async Prediction



```python

import aiohttp

import asyncio



async def predict\_async(file\_path):

&nbsp;   url = "http://localhost:8000/predict"

&nbsp;   

&nbsp;   async with aiohttp.ClientSession() as session:

&nbsp;       with open(file\_path, 'rb') as f:

&nbsp;           data = aiohttp.FormData()

&nbsp;           data.add\_field('file', f, filename='scan.nii.gz')

&nbsp;           

&nbsp;           async with session.post(url, data=data) as response:

&nbsp;               return await response.json()



\# Usage

result = asyncio.run(predict\_async("scan.nii.gz"))

```



\### JavaScript/Node.js



```javascript

const axios = require('axios');

const FormData = require('form-data');

const fs = require('fs');



async function predictAlzheimers(filePath) {

&nbsp;   const url = 'http://localhost:8000/predict';

&nbsp;   

&nbsp;   const formData = new FormData();

&nbsp;   formData.append('file', fs.createReadStream(filePath));

&nbsp;   

&nbsp;   try {

&nbsp;       const response = await axios.post(url, formData, {

&nbsp;           headers: formData.getHeaders()

&nbsp;       });

&nbsp;       

&nbsp;       console.log('Prediction:', response.data.predicted\_class);

&nbsp;       console.log('Confidence:', response.data.confidence);

&nbsp;       return response.data;

&nbsp;   } catch (error) {

&nbsp;       console.error('Error:', error.response.data.detail);

&nbsp;       return null;

&nbsp;   }

}



// Usage

predictAlzheimers('path/to/scan.nii.gz');

```



\### cURL



```bash

\# Simple prediction

curl -X POST "http://localhost:8000/predict" \\

&nbsp; -H "accept: application/json" \\

&nbsp; -F "file=@scan.nii.gz"



\# With authentication (production)

curl -X POST "https://api.example.com/predict" \\

&nbsp; -H "Authorization: Bearer YOUR\_TOKEN" \\

&nbsp; -F "file=@scan.nii.gz"



\# Batch prediction

curl -X POST "http://localhost:8000/predict/batch" \\

&nbsp; -F "files=@scan1.nii.gz" \\

&nbsp; -F "files=@scan2.nii.gz" \\

&nbsp; -F "files=@scan3.nii.gz"



\# Get explanation

curl -X POST "http://localhost:8000/predict/explain" \\

&nbsp; -F "file=@scan.nii.gz"

```



\### R



```r

library(httr)



predict\_alzheimers <- function(file\_path) {

&nbsp; url <- "http://localhost:8000/predict"

&nbsp; 

&nbsp; response <- POST(

&nbsp;   url,

&nbsp;   body = list(file = upload\_file(file\_path)),

&nbsp;   encode = "multipart"

&nbsp; )

&nbsp; 

&nbsp; if (status\_code(response) == 200) {

&nbsp;   result <- content(response)

&nbsp;   cat("Prediction:", result$predicted\_class, "\\n")

&nbsp;   cat("Confidence:", result$confidence, "\\n")

&nbsp;   return(result)

&nbsp; } else {

&nbsp;   cat("Error:", content(response)$detail, "\\n")

&nbsp;   return(NULL)

&nbsp; }

}



\# Usage

result <- predict\_alzheimers("path/to/scan.nii.gz")

```



---



\## Interactive Documentation



\### Swagger UI



Access interactive API documentation at:

```

http://localhost:8000/docs

```



Features:

\- Try out API endpoints directly

\- View request/response schemas

\- Download OpenAPI specification



\### ReDoc



Alternative documentation at:

```

http://localhost:8000/redoc

```



Features:

\- Clean, modern interface

\- Better for reading documentation

\- Searchable



---



\## Performance Tips



\### 1. Batch Processing



Use batch endpoint for multiple files:

```python

\# ❌ Slow: Multiple requests

for file in files:

&nbsp;   predict(file)



\# ✅ Fast: Single batch request

predict\_batch(files)

```



\### 2. Async Requests



Use async for concurrent predictions:

```python

\# Process multiple files concurrently

tasks = \[predict\_async(f) for f in files]

results = await asyncio.gather(\*tasks)

```



\### 3. Connection Pooling



Reuse HTTP connections:

```python

session = requests.Session()

for file in files:

&nbsp;   session.post(url, files={'file': open(file, 'rb')})

```



\### 4. File Compression



Use .nii.gz instead of .nii:

\- Smaller file size

\- Faster upload

\- Same accuracy



---



\## Troubleshooting



\### Common Issues



\*\*Issue:\*\* "Model not loaded" error

```

Solution: Wait for model to load (check /health endpoint)

```



\*\*Issue:\*\* Slow predictions

```

Solution: 

\- Use batch endpoint for multiple files

\- Check server resources (CPU/Memory)

\- Consider GPU deployment

```



\*\*Issue:\*\* "Invalid file format" error

```

Solution: 

\- Ensure file is .nii or .nii.gz

\- Check file is not corrupted

\- Verify file can be opened with nibabel

```



---



\## Support



For API issues or questions:

\- \*\*GitHub Issues:\*\* https://github.com/yourusername/alzheimers-mri-setnn/issues

\- \*\*Email:\*\* your.email@example.com

\- \*\*Documentation:\*\* https://github.com/yourusername/alzheimers-mri-setnn/docs

