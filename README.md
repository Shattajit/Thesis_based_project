# Smart Attendance System Using Machine Learning And Deep Learning

## Overview

This project implements a face recognition system that detects faces in images, compares them with a predefined dataset, and logs attendance. It uses MTCNN for face detection, FaceNet (InceptionResnetV1) for facial feature extraction, and cosine similarity for comparing embeddings. The system logs the attendance in an `attendance.json` file and generates a report in `attendance_log.xlsx`. The detected faces are also saved, and their details are logged in `detected_faces.txt`.

## Features

- **User Authentication**: Register, login, and update profiles. Secure routes with JWT and role-based access.
  
- **Category Management**: Create, update, delete, and list product categories.
  
- **Product Management**: CRUD operations for products, including image upload, filtering, and pagination.
  
- **Order Management**: Place orders, view order details, and update order status.
  
- **Payment Integration**: Process payments using Braintree.
  
- **Search and Filters**: Search products by keyword and filter by category and price range.

  

## Technologies Used

- **Frontend**: React 
  
- **Backend**: Node.js, Express
  
- **Database**: MongoDB
  
- **Authentication**: JSON Web Tokens (JWT)
  
- **Payment Gateway**: Braintree
  
- **File Upload**: Formidable
  
- **Environment Variables**: dotenv

## Getting Started

### Prerequisites

- Node.js and npm
  
- MongoDB
  
- Braintree account (for payment processing)

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd <project-directory>

2. **Install dependencies**

   ```bash
   npm install
   
3. **Create a .env file**
   
   Create a .env file in the root directory and add the following environment variables:

   ```bash
   PORT=<your-port>
   MONGO_URI=<your-mongodb-uri>
   JWT_SECRET=<your-jwt-secret>
   BRAINTREE_MERCHANT_ID=<your-braintree-merchant-id>
   BRAINTREE_PUBLIC_KEY=<your-braintree-public-key>
   BRAINTREE_PRIVATE_KEY=<your-braintree-private-key>

5. **Run the application**

   ```bash
   npm start
   
The server will be running on http://localhost:8080 (or your specified port).



## API Endpoints

Authentication

- POST `/api/v1/auth/register`: Register a new user.
  
- POST `/api/v1/auth/login`: Log in a user.
  
- POST `/api/v1/auth/forgot-password`: Request a password reset.

  

Categories

- POST `/api/v1/category/create-category`: Create a new category.

- PUT `/api/v1/category/update-category/:id`: Update an existing category.

- GET `/api/v1/category/get-category`: Get all categories.

- GET `/api/v1/category/single-category/:slug`: Get a single category by slug.

- DELETE `/api/v1/category/delete-category/:id`: Delete a category.



Products

- POST `/api/v1/product/create-product`: Create a new product.

- PUT `/api/v1/product/update-product/:pid`: Update an existing product.

- GET `/api/v1/product/get-product`: Get all products.

- GET `/api/v1/product/get-product/:slug`: Get a single product by slug.

- GET `/api/v1/product/product-photo/:pid`: Get a product's photo.

- DELETE `/api/v1/product/delete-product/:pid`: Delete a product.

- POST `/api/v1/product/product-filters`: Filter products by category and price.

- GET `/api/v1/product/product-count`: Get the total count of products.

- GET `/api/v1/product/product-list/:page`: Get products with pagination.

- GET `/api/v1/product/search/:keyword`: Search products by keyword.

- GET `/api/v1/product/related-product/:pid/:cid`: Get related products.

- GET `/api/v1/product/product-category/:slug`: Get products by category.



Orders

- GET `/api/v1/auth/orders`: Get orders for the logged-in user.

- GET `/api/v1/auth/all-orders`: Get all orders (admin only).

- PUT `/api/v1/auth/order-status/:orderId`: Update the status of an order.



Payment

- GET `/api/v1/product/braintree/token`: Get Braintree payment token.

- POST `/api/v1/product/braintree/payment`: Process Braintree payment.



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

