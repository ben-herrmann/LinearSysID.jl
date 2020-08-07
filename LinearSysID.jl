module LinearSysID

export Hankel, ERA, OKID, BPOD, balred, baltransform

using LinearAlgebra, ControlSystems

function Hankel(Y::Array{<:Number,1})
    nH = Int(ceil((size(Y,1)/2)))
    H = []
    for i=1:nH
        push!(H,hcat([Y[j+i-1,:,:] for j=1:nH]...))
    end
    H = vcat(H...)
    return H
end

function Hankel(Y::Array{<:Number,3})
    nH = Int(ceil((size(Y,1)/2)))
    H = []
    for i=1:nH
        push!(H,hcat([Y[j+i-1,:,:] for j=1:nH]...))
    end
    H = vcat(H...)
    return H
end

function ERA(YY::Array{<:Number,1}, r::Integer)
    D = YY[1,:,:]
    ns, nu = size(D)
    H = Hankel(YY[2:end-1,:,:])
    H2 = Hankel(YY[3:end,:,:])
    U,σ,V = svd(H, full=false)
    Ũ = U[:,1:r]
    Σ̃ = Diagonal(σ[1:r])
    Ṽ = V[:,1:r]
    A = Σ̃^(-1/2)*Ũ'*H2*Ṽ*Σ̃^(-1/2)
    B = Σ̃^(-1/2)*Ũ'*H[:,1:nu]
    C = H[1:ns,:]*Ṽ*Σ̃^(-1/2)
    sysERA = ss(A,B,C,D,1)
    return sysERA
end

function ERA(YY::Array{<:Number,3}, r::Integer)
    D = YY[1,:,:]
    ns, nu = size(D)
    H = Hankel(YY[2:end-1,:,:])
    H2 = Hankel(YY[3:end,:,:])
    U,σ,V = svd(H, full=false)
    Ũ = U[:,1:r]
    Σ̃ = Diagonal(σ[1:r])
    Ṽ = V[:,1:r]
    A = Σ̃^(-1/2)*Ũ'*H2*Ṽ*Σ̃^(-1/2)
    B = Σ̃^(-1/2)*Ũ'*H[:,1:nu]
    C = H[1:ns,:]*Ṽ*Σ̃^(-1/2)
    sysERA = ss(A,B,C,D,1)
    return sysERA
end

function OKID(Y::Array{<:Number,2}, U::Array{<:Number,2}, nδ::Integer)
    ny, nt = size(Y)
    nu = size(U,1)
    V = Array{Float64}(undef, nu+(nu+ny)*nδ, nt)
    UY = [U; Y]
    V[1:nu,:] = U
    for i=1:nδ
        V[nu+1+(nu+ny)*(i-1):nu+(nu+ny)*i, :] = [zeros(nu+ny,i) UY[:, 1:nt-i]]
    end
    Ȳ = Y/V
    D = Ȳ[:,1:nu]
    Ȳ1 = cat([Ȳ[:, nu+1+(nu+ny)*(i-1):nu+(nu+ny)*(i-1)+nu] for i=1:nδ]..., dims=3)
    Ȳ2 = cat([Ȳ[:, nu+1+(nu+ny)*(i-1)+nu:nu+(nu+ny)*i] for i=1:nδ]..., dims=3)
    YY =Array{Float64}(undef, nδ, ny, nu)
    YY[1,:,:] = Ȳ1[:,:,1] .+ Ȳ2[:,:,1]*D
    for k=2:nδ
        YY[k,:,:] = Ȳ1[:,:,k] .+ Ȳ2[:,:,k]*D
        for i=1:k-1
            YY[k,:,:] += Ȳ2[:,:,i]*YY[k-i,:,:]
        end
    end
    return [reshape(D,(1,size(D)...)); YY]
end

function OKID(Y::Array{Array{Float64,2}}, U::Array{Array{Float64,2}}, nδ::Integer)
    ne = length(Y)
    ny, nt = size(Y[1])
    nu = size(U[1],1)
    V = Array{Float64}(undef, nu+(nu+ny)*nδ, nt*ne)
    V[1:nu,:] = hcat(U...)
    for e=1:ne
        UY = [U[e]; Y[e]]
        for i=1:nδ
            Ve[nu+1+(nu+ny)*(i-1):nu+(nu+ny)*i, 1+nt*(e-1):nt+nt*(e-1)] = [zeros(nu+ny,i) UY[:, 1:nt-i]]
        end
    end
    Ȳ = hcat(Y...)/V
    D = Ȳ[:,1:nu]
    Ȳ1 = cat([Ȳ[:, nu+1+(nu+ny)*(i-1):nu+(nu+ny)*(i-1)+nu] for i=1:nδ]..., dims=3)
    Ȳ2 = cat([Ȳ[:, nu+1+(nu+ny)*(i-1)+nu:nu+(nu+ny)*i] for i=1:nδ]..., dims=3)
    YY =Array{Float64}(undef, nδ, ny, nu)
    YY[1,:,:] = Ȳ1[:,:,1] .+ Ȳ2[:,:,1]*D
    for k=2:nδ
        YY[k,:,:] = Ȳ1[:,:,k] .+ Ȳ2[:,:,k]*D
        for i=1:k-1
            YY[k,:,:] += Ȳ2[:,:,i]*YY[k-i,:,:]
        end
    end
    return [reshape(D,(1,size(D)...)); YY]
end

function BPOD(sys::StateSpace{},t::Array{<:Number,1},r::Integer)
    nt = length(t)
    nx,nu = size(sys.B)
    ny,~ = size(sys.C)
    sysAdj = ss(sys.A',sys.C',sys.B',sys.D')
    yDir,~,xDir = impulse(sysd,t)
    yAdj,~,xAdj = impulse(sysAdj,t)
    Ctr = []
    Obs = []
    for i=2:nt
        push!(Ctr, xDir[i,:,:])
        push!(Obs, xAdj[i,:,:]')
    end
    Ctr = hcat(Ctr...)
    Obs = vcat(Obs...)
    Hkl = Obs*Ctr
    U,σ,V = svd(Hkl, full=false)
    Ũ = U[:,1:r]
    Σ̃ = Diagonal(σ[1:r])
    Ṽ = V[:,1:r]
    Ψ = Ctr*Ṽ*Σ̃^(-1/2) # direct modes
    Φ = Obs'*Ũ*Σ̃^(-1/2) # direct modes
    Ã = Φ'*sys.A*Ψ
    B̃ = Φ'*sys.B
    C̃ = sys.C*Ψ
    D̃ = yDir[1,:,:]
    sysBPOD = ss(Ã,B̃,C̃,D̃,dt)
    return sysBPOD
end

function baltransform(sys::StateSpace)
    Wc = gram(sys,:c)
    Wo = gram(sys,:o)
    Tu = eigvecs(Wc*Wo)
    invTu = inv(Tu)
    Σc = invTu*Wc*invTu'
    Σo = Tu'*Wo*Tu
    Σs = Σc^(1/4)*Σo^(-1/4)
    T = Tu*Σs
    invT = inv(T)
     return T, invT
end

function balred(sys::StateSpace, r::Integer)
    T, invT = baltransform(sys)
    Tr = T[:,1:r]
    invTr = invT[1:r,:]
    sysr = ss(invTr*sys.A*Tr,invTr*sys.B,sys.C*Tr,sys.D, sys.Ts)
    return sysr, Tr, invTr
end

end # module
