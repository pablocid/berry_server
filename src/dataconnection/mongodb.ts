// import { Schema, Connection, connect as Connect, createConnection, Model, Document, PromiseProvider, Promise, ConnectionOptions, ConnectionOptionsBase } from "mongoose";
// import mongoose = require('mongoose');
// var config =  require('../../config/enviroment');

// export class MongoConnection{
//     public static STRING_CONNECTION: string = config.mongo.uri;
//     public MConnOptions:ConnectionOptionsBase;
//     private _connection: Connection;
//     public model: Model<Document>;

//     constructor(name:string, schm:Schema){
//         this.MConnOptions = {}
//         //use q promises
//         global.Promise = require("q").Promise;
//         //use q library for mongoose promise
//         mongoose.Promise = global.Promise;
//         this._connection =  createConnection(MongoConnection.STRING_CONNECTION);
//         this.model = this._connection.model(name, schm);
//     }
// }