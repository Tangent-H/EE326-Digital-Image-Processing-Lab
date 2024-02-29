> 其实终究还是官方源最靠谱最稳定，镜像源动不动还是会出不少问题，还是尽量用官方源，但是由于国内网络环境，需要对`conda`和`pip`设置代理

1. 清空当前默认的`channel`（如果是国内源的话）

   ```
   conda config --remove-key channels
   ```

2. 改conda的proxy为`clash-verge`的系统代理

   ```
   conda config --set proxy_servers.http http://127.0.0.1:7897
   conda config --set proxy_servers.https http://127.0.0.1:7897
   ```

   

如果需要清除代理的设置，则执行下面这个

```
conda config --remove proxy_servers.http
conda config --remove proxy_servers.https
```

查看

```
conda config --show
```



`pip`同样可以设置代理

```
pip config set global.proxy http://127.0.0.1:7897
```

如果要清除，则用

```
pip config unset global.proxy
```

查看

```
pip config list
```

