

default-rule: PythonExtrasC/PythonExtrasC.cpp
	mkdir -p x64/Release
	g++ -Wall -g -O2 -std=c++14 -fPIC -c PythonExtrasCLib/PythonExtrasCLib.cpp -o x64/Release/PythonExtrasCLib.o
	ar rvs x64/Release/libPythonExtrasCLib.a x64/Release/PythonExtrasCLib.o
	g++ -Wall -g -O2 -std=c++14 -fPIC -I ./PythonExtrasCLib -c PythonExtrasC/BufferedNdArrayWrapper.cpp -o x64/Release/BufferedNdArrayWrapper.o
	g++ -Wall -g -O2 -std=c++14 -fPIC -I ./PythonExtrasCLib -c PythonExtrasC/MtPatchExtractorWrapper.cpp -o x64/Release/MtPatchExtractorWrapper.o
	g++ -Wall -g -O2 -std=c++14 -fPIC -I ./PythonExtrasCLib -c PythonExtrasC/PythonExtrasC.cpp -o x64/Release/PythonExtrasC.o
	g++ -Wall -g -O2 -std=c++14 -fPIC -I ./PythonExtrasCLib -c PythonExtrasC/RenderingTools.cpp -o x64/Release/RenderingTools.o
	g++ -Wall -g -shared -o x64/Release/PythonExtrasC.so x64/Release/PythonExtrasCLib.o x64/Release/BufferedNdArrayWrapper.o x64/Release/MtPatchExtractorWrapper.o x64/Release/RenderingTools.o x64/Release/PythonExtrasC.o
	cp x64/Release/PythonExtrasC.so ../PythonExtras/c_dll/
#	g++ -Wall -g -O2 -std=c++14 -fPIC -I ./PythonExtrasCLib -c Profile/Profile.cpp -o x64/Release/Profile.o
#	g++ -Wall -g -pthread -o x64/Release/Profile x64/Release/Profile.o x64/Release/PythonExtrasC.so

