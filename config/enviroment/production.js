'use strict';

// Production specific configuration
// =================================
module.exports = {
    // MongoDB connection options
    mongo: {
        uri: process.env.MONGODB_URI ||
            process.env.MONGOHQ_URL ||
            process.env.MONGODB_LOCAL ||
            'mongodb://localhost/default'
    }
};