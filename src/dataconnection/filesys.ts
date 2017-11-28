import * as Q from 'q';
import { readFile as ReadFile, writeFile as WriteFile } from 'fs';


export class FileSysConnection {
    constructor() { }

    public getJSON(value: string) {
        value = this._setJSONExtension(value);
        var def = Q.defer();

        if (!value) { def.reject({ error: "there is no path" }) }

        let path = __dirname + '/../../JSON/' + value;

        ReadFile(path, (err, data) => {
            if (err) { def.reject({ error: "Problem with getting the file", server: err }); }
            def.resolve(JSON.parse(data.toString()));
        });

        return def.promise;
    }

    public setJSON(name: string, data: JSON) {
        name = this._setJSONExtension(name);
        var def = Q.defer();
        if (!name || !data) { def.reject({ error: "there is no path" }) }
        let dataString = JSON.stringify(data);
        let path = __dirname + '/../../JSON/' + name;

        WriteFile(path, dataString, { encoding: 'utf8' }, (err) => {
            if (err) { def.reject({ error: "Problem with writing file", server: err }); }
            def.resolve({ status: 'OK' });
        });

        return def.promise;
    }

    private _setJSONExtension(value: string): string {
        let ext = value.split('.').pop();
        if (ext !== 'json') {
            return value + '.json';
        } else {
            return value;
        }
    }
}