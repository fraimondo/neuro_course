##

# Instalar WSL

1. Abrir una terminal y ejecutar:

```
wsl --install
```

Va a tardar un rato. Al finalizar, reiniciar el sistema.

2. Abrir la aplicación "WSL", va a pedir crear un usuario y un pass. Elegir algo básico como `test` y `test`.


# Instalar VSCode

1. Instalar VSCode

2. Instalar la extension "Remote Development" y "WSL".

3. Abrir la extension Remote y elegir en "WSL Targets", la distro de ubuntu. Click en el icono de la flecha para conectarse a la sesion WSL.

4. View -> Terminal

# Instalar miniconda

1. En la terminal abierta de VSCode, ejecutar

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
 
2. Ejecutar

```
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

3. Escribir "yes" en los dos prompts.

4. Cerrar la terminal.

5. Volver a abrir la terminal.

6. Ejecutar

```
conda create -n neurodc python=3.11
```

7. Activar el environment

```
conda activate neurodc
```

# Instalar datalad
1. Ejecutar

```
conda install -c conda-forge datalad
```

2. Ejecutar

```
git config --global user.email test@test.com
git config --global user.name test
```

# Instalar junifer
1. Ejecutar

```
pip install junifer
```

# Instalar Docker Desktop

1. Descargar docker desktop de https://www.docker.com/products/docker-desktop/

2. Instalar. Elegir usar WSL. 

3. Reiniciar

4. Abrir docker desktop

5. Opcional (log in/crear cuenta)

6. Ir a settings (icono arriba a la derecha) -> Resources -> WSL Integration -> Enable integration with Ubuntu. 

7. Apply and restart.

8. Reiniciar el PC.


# Continuar con los containers para junifer

1. Abrir VScode (se deberia abrir directamente en el WSL)

2. Instalar ANTS de las dependencias opcionales de Junifer: 

https://juaml.github.io/junifer/main/installation.html#installing-external-dependencies

