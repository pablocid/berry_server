
export class TopBar {
    public title:string;
    public labels:Object;
    constructor(){
        this.title = "Plant Biotech";
        this.labels = {
            login:'logearse',
            languaje:'Lenguaje',
            lang:['English','Español'],
            adminArea:'Área de administración'
        }

    }

    get data(){
        return {
            title:'Lab de Biotech'
        }
    }

}