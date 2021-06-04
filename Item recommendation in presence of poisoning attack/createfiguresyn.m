function createfiguresyn(X1, YMatrix1)
%CREATEFIGURE(X1, YMatrix1)
%  X1:  vector of x data
%  YMATRIX1:  matrix of y data

%  Auto-generated by MATLAB on 23-May-2021 09:22:51

% Create figure
figure1 = figure;

% Create axes
axes1 = axes('Parent',figure1,...
    'Position',[0.13 0.121432926829268 0.775 0.803567073170732]);
hold(axes1,'on');

% Create multiple lines using matrix input to plot
plot1 = plot(X1,YMatrix1,...
    'MarkerIndices',[1 11 21 31 41 51 61 71 81 91 101 111 121 131 141 151 161 171 181 191],...
    'LineWidth',3,...
    'Parent',axes1);
set(plot1(1),'DisplayName','GradientGreedy','Marker','x');
set(plot1(2),'DisplayName','Gradient RepGreedy','MarkerSize',8,'Marker','o',...
    'LineStyle','--');
set(plot1(3),'DisplayName','ExtraGradient Greedy','MarkerSize',8,...
    'Marker','diamond');
set(plot1(4),'DisplayName','ExtraGradient RepGreedy','LineStyle','-.',...
    'MarkerIndices',[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191]);

% Create ylabel
ylabel('\phi(x)','FontName','Times New Roman');

% Create xlabel
xlabel('Iteration','FontName','Times New Roman');

box(axes1,'on');
grid(axes1,'on');
% Set the remaining axes properties
set(axes1,'FontName','Times New Roman','FontSize',30);
% Create legend
legend1 = legend(axes1,'show');
set(legend1,...
    'Position',[0.590434782608697 0.65625 0.281739130434783 0.194359756097561]);
