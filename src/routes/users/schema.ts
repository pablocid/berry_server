import { Schema } from "mongoose";

export var UserSchema: Schema = new Schema({
    created: Schema.Types.Date,
    email: Schema.Types.String,
    password: Schema.Types.String,
    firstName: Schema.Types.String,
    lastName: Schema.Types.String,
    role: Schema.Types.String,
});