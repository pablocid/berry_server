import { BaseRoute } from '../../models/class.route'
import { NextFunction, Request, Response, Router } from 'express';
import { exec } from 'child_process';

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
        console.log('Comenzando el routing')
        r.get("/", (req: Request, res: Response, next: NextFunction) => {
            console.log('get in berry analyzaer')
            obj.index(req, res, next);
        });
        // r.get("/:name", (req: Request, res: Response, next: NextFunction) => {
        //     obj.partial(req, res, next);
        // });
        // r.post("/:name", (req: Request, res: Response, next: NextFunction) => {
        //     console.log('partial name post ')
        //     obj.update(req, res, next);
        // });

        return r;
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


        exec(`python img-scripts/transform.py bayas.jpg`, function (err, stdout, stderr) {
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
            res.json(datos)
            console.dir(datos)
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

    /**
     *
     * @class BerryAnalyzerRoute
     * @method partial
     * @param req {Request} The express Request object.
     * @param res {Response} The express Response object.
     * @next {NextFunction} Execute the next method.
     */
    public partial(req: Request, res: Response, next: NextFunction) {
        let name = req.params.name;
    }


}