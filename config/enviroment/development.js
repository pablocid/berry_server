'use strict';

// Development specific configuration
// ==================================
module.exports = {

    // MongoDB connection options
    mongo: {
        uri: process.env.MONGODB_LOCAL || 'mongodb://localhost/default'
    }

};