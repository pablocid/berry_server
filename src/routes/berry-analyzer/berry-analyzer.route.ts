import { BaseRoute } from '../../models/class.route'
import { NextFunction, Request, Response, Router } from 'express';
import { exec } from 'child_process';
import { readFileSync, unlink } from 'fs';
import * as Multer from 'multer';


interface OpenCV {
    readImage(path: string, callback: any): void;
}
const cv = <OpenCV>require('opencv');
export class BerryAnalyzerRoute extends BaseRoute {
    /**
     * Create the routes.
     *
     * @class DataPartialRoute
     * @method route
     * @static
     */
    public static get route() {
        let r = Router();
        var obj = new BerryAnalyzerRoute();

        r.get("/", (req: Request, res: Response, next: NextFunction) => {
            console.log('get in berry analyzaer')
            obj.index(req, res, next);
        });
        // r.get("/:name", (req: Request, res: Response, next: NextFunction) => {
        //     obj.partial(req, res, next);
        // });
        r.post("/upload", BerryAnalyzerRoute.multer('archivo'), (req: Request, res: Response, next: NextFunction) => {
            obj.upload(req, res, next);
        });

        return r;
    }

    public static multer(value: string) {
        const opt: Multer.Options = { dest: 'asdf' }
        return Multer({ dest: 'uploads/' }).single(value)

    }

    /**
     * Constructor
     *
     * @class BerryAnalyzerRoute
     * @constructor
     */
    constructor() {
        super();
    }

    /**
     *
     * @class BerryAnalyzerRoute
     * @method index
     * @param req {Request} The express Request object.
     * @param res {Response} The express Response object.
     * @next {NextFunction} Execute the next method.
     */
    public index(req: Request, res: Response, next: NextFunction) {
        //lista de los routes disponibles


        exec(`python img-scripts/transform.py bayas.jpg`, (err, stdout, stderr) => {
            if (stderr) { console.dir(stderr); return; }

            //console.dir(stdout.split('\n'));
            const path = stdout.split('\n')[0]
            const data = stdout.split('\n')[1]
            //console.dir(path)
            //console.dir(JSON.parse(data))


            const datos = (<Array<number[]>>JSON.parse(data)).map(x => {
                return {
                    width: x[0],
                    height: x[1],
                    mean: x[2],
                    area: x[3],
                    rectProp: x[4],
                    circleProp: x[5],
                    ellipseProp: x[6]
                }
            })

            res.json({ url: this._base64_encode(path), data: datos })

            this._removeTmpImg(path);
        })
        // console.log(process.env.PWD)
        // cv.readImage(process.env.PWD + '/uploads/bayas.jpg', function (err: any, im: any) {
        //     im.convertGrayscale()
        //     im.canny(5, 300)
        //     im.houghLinesP()
        //     im.save(process.env.PWD + '/uploads/out.jpg');
        // })
        // res.json({ lista: [1, 2, 3, 4, 5] });
    }

    private _base64_encode(file: string) {
        // read binary data
        var bitmap = readFileSync(file);
        // convert binary data to base64 encoded string
        return new Buffer(bitmap).toString('base64');
    }

    private _removeTmpImg(path: string) {
        unlink(path, (err) => {

        })
    }

    /**
     *
     * @class BerryAnalyzerRoute
     * @method partial
     * @param req {Request} The express Request object.
     * @param res {Response} The express Response object.
     * @next {NextFunction} Execute the next method.
     */
    public upload(req: Request, res: Response, next: NextFunction) {

        exec(`python img-scripts/transform.py ${req.file.path}`, (err, stdout, stderr) => {
            if (stderr) { 
                console.dir(stderr); 
                res.json({ error:stderr })
                unlink(req.file.path, (err) => { })
                return; 
            }

            const path = stdout.split('\n')[0]
            if(path === '001'){
                res.status(500).json({message:"no se pudo ajustar la perspectiva de la imagen"})
                return;
            }
            const data = stdout.split('\n')[1]
            const datos = (<Array<number[]>>JSON.parse(data)).map(x => {
                return {
                    width: x[0],
                    height: x[1],
                    mean: x[2],
                    area: x[3],
                    rectProp: x[4],
                    circleProp: x[5],
                    ellipseProp: x[6]
                }
            })

            res.json({ url: this._base64_encode(path), data: datos })

            this._removeTmpImg(path);
            this._removeTmpImg(req.file.path);
        })
        // res.json({
        //     path: req.file.path,
        //     originalName: req.file.originalname,
        //     encoding: req.file.encoding,
        //     minetype: req.file.mimetype,
        //     size: req.file.size,
        //     destination: req.file.destination,
        //     filename: req.file.filename,
        //     fieldname: req.file.fieldname,
        //     buffer: req.file.buffer
        // });

    }


}