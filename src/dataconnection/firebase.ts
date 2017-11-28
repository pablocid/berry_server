import { defer } from 'q';
import { initializeApp, database, credential } from "firebase-admin";

var serviceAccount = require("../../firekey.json");

initializeApp({
    credential: credential.cert(serviceAccount),
    databaseURL: "https://myexpress-88f9a.firebaseio.com"
});
// TODO: Generalizar la conexión

export class FireConnection {
    public db: database.Database;
    private _init: any;
    constructor() {
        this.db = database();
    }

    getSome() {
        var def = defer();
        this.db.ref('someshit/krap').set({
            crap:'popo',
            pi:'pichi'
        }).then(x=>{
            console.log(x);
            def.resolve(x)
        })
        return def.promise;

    }
}