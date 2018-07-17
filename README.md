> ```
>            _                 _ _  _____ _    _ _____          
>           | |               | (_)/ ____| |  | |  __ \   /\    
>   ___ ___ | |_ __   ___   __| |_| |    | |  | | |  | | /  \   
>  / __/ _ \| | '_ \ / _ \ / _` | | |    | |  | | |  | |/ /\ \  
> | (_| (_) | | |_) | (_) | (_| | | |____| |__| | |__| / ____ \ 
>  \___\___/|_| .__/ \___/ \__,_|_|\_____|\____/|_____/_/    \_\
>             | |                                               
>             |_|                                               
> ```  
    
----- LIBRERIA PER OPTION PRICING ----- PRODUZIONE ARTIGIANALE -----

PER USARE IL PROGRAMMA:
Il makefile crea due oggetti eseguibili: "pricer" e "pricer_comp", il primo sfrutta unicamente la GPU, il secondo esegue il confronto tra i risultati ottenuti su CPU e GPU.
Il file contenente i dati di input viene specificato nella funziore "Reader" nel file Utilities.cu (di default è impostato il file "input.conf" nella cartella "DATA"). 

Ultimo aggiornamento: sa dio
