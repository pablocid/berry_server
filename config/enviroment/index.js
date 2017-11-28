"use strict";

var _ = require('lodash');

// All configurations will extend these options
// ============================================
var all = {
    env: process.env.NODE_ENV,

    // Server port
    port: process.env.PORT || 8000,

    // Server IP
    ip: process.env.IP || '0.0.0.0',

    // Secret for session, you will want to change this and make it an environment variable
    secrets: {
        session: process.env.SECRET_SESSION || 'LSKJLKFJLKSDJFLKSJDLKFJLDSJFLKDJ',
        cookieParser: process.env.COOKIEPARSER || '',
        app: process.env.APP_SECRET || 'LKJLJLJLKJDLKFJLJLFJDSJFLjkl213123l12'
    },

    //base de datos a utilizar
    database: process.env.DATABASE || 'mongodb',

    // MongoDB connection options
    mongo: {
        options: {
            db: {
                safe: true
            }
        }
    },

    google: {
        clientID: process.env.GOOGLE_ID || 'id',
        clientSecret: process.env.GOOGLE_SECRET || 'secret',
        callbackURL: (process.env.DOMAIN || '') + '/auth/google/callback'
    },
    AWS: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID || 'id',
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || 'secret'
    }
};

// Export the config object based on the NODE_ENV
// ==============================================
module.exports = _.merge(
    all,
    require('./shared'),
    require('./' + process.env.NODE_ENV + '.js') || {});